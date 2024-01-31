import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.onnx
from prettytable import PrettyTable

from config.env_config import EnvironmentConfig
from example_datasets.ptb.ptb_args import args
from example_datasets.ptb.ptb_dataloader import PTBDataLoader
from example_datasets.ptb.ptb_model_wrapper import RNNModel, InvalidModelConfigurationException
from model.circular_reference_exception import CircularReferenceException
from model.trainer_interface import TrainerInterface
from ops.block_state_builder import BlockStateBuilder
from ops.nas_controller import NASController
from persistence.model_persistence import ModelPersistence
from utils.device_controller import DeviceController
from utils.logger import LOG
from utils.random_generator import RandomGenerator
from utils.slack_post import SlackPost
from utils.tensorboard_writer import TensorBoardWriter

LOG = LOG.get_instance().get_logger()


alpha = 2
beta = 1

GOOGLE_DRIVE_EXISTS = os.path.exists('/content/drive/My Drive/msc_run')
OUTPUT = f'./output' if not GOOGLE_DRIVE_EXISTS else f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}/output'

def get_batch(source, i, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()


class PtbTrainer(TrainerInterface):

    def __init__(self, architecture, model_identifier, epochs, nlayers=None, best_ppl=None):
        self.model_architecture = architecture
        self.model_identifier = model_identifier
        self.eval_batch_size = 10
        self.interval = 200  # interval to report
        self.epochs = epochs
        self.epoch_loss_nan_count = 0
        self.best_ppl = best_ppl
        self.nlayers = nlayers
        self.model_params = 10.0e+10

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print('Ptb_Trainer deleted.')

    def build_model(self, post_slack=False):
        num_layers = self.nlayers if self.nlayers is not None else args.nlayers
        return RNNModel(self.model_architecture, self.model_identifier, PTBDataLoader.get_ntokens(), args.emsize,
                        args.nhid, num_layers, dropout=args.dropout, post_slack=post_slack)

    def run(self, parent_epoch_performance, model=None, force_training=False, warm_start_parent=None,
            opt_override=None, override_train_data=False):

        if EnvironmentConfig.get_config('simulate_results'):
            loss = RandomGenerator.uniform(1, 6.5)
            MIN_BLOCKS = len(BlockStateBuilder.get_basic_architecture().blocks.keys())
            MAX_BLOCKS = len(BlockStateBuilder.get_lstm_architecture().blocks.keys())
            self.model_params = RandomGenerator.randint(MIN_BLOCKS, MAX_BLOCKS)
            return loss, math.exp(loss), {}

        LOG.info(f'Running for {self.model_identifier} with {self.epochs} epochs.')

        if model is None:
            try:
                model = self.build_model(post_slack=True)
            except CircularReferenceException:
                return float("inf"), float("inf"), {}

            if warm_start_parent is not None:
                try:
                    if os.path.exists(f'{OUTPUT}/models/{warm_start_parent}.tar'):
                        ModelPersistence.load_model(f'{warm_start_parent}', model)
                        LOG.info(f'Warm started {self.model_identifier} from parent {warm_start_parent}.')
                    elif os.path.exists(f'{OUTPUT}/models/{warm_start_parent}_ptb.tar'):
                        ModelPersistence.load_model(f'{warm_start_parent}_ptb', model)
                        LOG.info(f'Warm started {self.model_identifier} from parent {warm_start_parent}.')
                    else:
                        LOG.info(
                            f'Warm start enabled for {self.model_identifier} but parent {warm_start_parent} could not be found.')
                        if os.path.exists(f'{OUTPUT}/models/{self.model_identifier}_ptb.tar'):
                            ModelPersistence.load_model(f'{self.model_identifier}_ptb', model)
                except Exception as e:
                    """
                    This exception is okay, it can happen that some input/output dimensions changed during network morphism,
                    and thus PyTorch will throw an exception to indicate that the model state dictionary cannot be restored.
                    """
                    error_msg = f'Unable to warm start {self.model_identifier} from parent {warm_start_parent}; exception = {e}.'
                    LOG.info(error_msg)
                    SlackPost.post_neutral('Warm start issue', error_msg)

        self.count_parameters(model)
        train_data, val_data, test_data = PTBDataLoader.get_data()
        self.batch_size = args.batch_size
        if override_train_data:
            train_data = copy.deepcopy(val_data)
            val_data = copy.deepcopy(test_data)
            self.batch_size = self.eval_batch_size

        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model.to(DeviceController.get_device())
            criterion.to(DeviceController.get_device())

        # Loop over epochs.
        lr = args.lr
        best_val_loss = None
        if opt_override is not None:
            opt_val = opt_override
        else:
            opt_val = args.opt

        LOG.info(f'Using {opt_val} optimizer.')
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        if opt_val == 'Adam':
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lr = 1e-3
        if opt_val == 'Momentum':
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
        if opt_val == 'RMSprop':
            opt = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
            lr = 0.001
        if opt_val == 'ASGD':
            opt = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=args.wdecay)

        LOG.info(opt)
        opt_override_txt = f'_{opt_override}' if opt_override else ""
        MODEL_CHECKPOINT_PATH = f'{OUTPUT}/ptb_checkpoint_{self.model_identifier}{opt_override_txt}.tar'
        model_epoch_performance = {'training_time': []}
        if not os.path.exists(f'{OUTPUT}/models/{self.model_identifier}_ptb.tar') or force_training:
            try:
                epoch = 0
                epoch_check = 0
                epochs_adjusted = False

                if os.path.exists(MODEL_CHECKPOINT_PATH):
                    checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
                    epoch = checkpoint['epoch']
                    epoch_check = checkpoint['epoch_check']
                    best_val_loss = checkpoint['best_val_loss']
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    LOG.info(f'Successfully loaded training for model {self.model_identifier} from checkpoint.')

                NASController.start_training_time_for_architecture(model.identifier)
                while epoch_check < self.epochs:
                    epoch_start_time = time.time()
                    self.train(model, epoch, lr, train_data, criterion, opt, parent_epoch_performance)

                    try:
                        val_loss = self.evaluate(model, val_data, criterion)
                        val_ppl = math.exp(val_loss)
                        LOG.info('-' * 89)
                        LOG.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                 'valid ppl {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                                            val_loss, val_ppl))
                        LOG.info('-' * 89)
                        model_epoch_performance['training_time'].append(time.time() - epoch_start_time)
                        if not best_val_loss or val_loss < best_val_loss:
                            val_loss_str = '{:5.4f}'.format(val_loss)
                            best_val_loss_str = 'None' if not best_val_loss else '{:5.4f}'.format(best_val_loss)
                            LOG.info(
                                f'Saved new model for {self.model_identifier} :: {val_loss_str} vs {best_val_loss_str}')
                            ModelPersistence.save_model(f'{self.model_identifier}{opt_override_txt}_ptb', model)
                            best_val_loss = val_loss
                        else:
                            # Anneal the learning rate if no improvement has been seen in the validation dataset.
                            if opt_val == 'SGD' or opt_val == 'Momentum':
                                lr /= 4.0
                                for group in opt.param_groups:
                                    group['lr'] = lr

                        scalar_prefix = 'PTB' if opt_override is None else f'PTB/{opt_val}'
                        TensorBoardWriter.write_scalar(self.model_identifier, f'{scalar_prefix}/Loss', val_loss, epoch)
                        TensorBoardWriter.write_scalar(self.model_identifier, f'{scalar_prefix}/PPL', val_ppl, epoch)

                        if self.best_ppl is not None and not epochs_adjusted and (self.epochs - epoch_check) <= 5:
                            diff = val_ppl - self.best_ppl
                            check_thresh = 100.0
                            if diff > 0:
                                check_thresh = (diff / self.best_ppl) * 100

                            if check_thresh < 25:  # 25% threshold
                                epochs_adjusted = True
                                epoch_diff = 5
                                if epoch_check - epoch_diff < 0:
                                    epoch_check = -1
                                else:
                                    epoch_check -= epoch_diff
                                LOG.info(f'Adjusted epochs for {self.model_identifier} with ppl diff of {diff}.')
                                check_thresh_str = '{:5.5f}'.format(check_thresh)
                                SlackPost.post_neutral('Epochs adjusted',
                                                        f'{self.model_identifier} achieved PPL threshold of {check_thresh_str} resulting in an epoch adjustment of {epoch_diff} -> {epoch_check}.')

                    except OverflowError:
                        LOG.info(f'OverflowError when evaluating {self.model_identifier}.')

                    epoch += 1
                    epoch_check += 1
                    torch.save({
                        'epoch': epoch,
                        'epoch_check': epoch_check,
                        'best_val_loss': best_val_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict()
                    }, MODEL_CHECKPOINT_PATH)

            except KeyboardInterrupt:
                LOG.info('-' * 89)
                LOG.info('Exiting from training early')
            except InvalidModelConfigurationException:
                LOG.info(f'Model {self.model_identifier} thrown invalid configuration exception.')
                self.handle_exception(MODEL_CHECKPOINT_PATH, model.identifier)
                return float("inf"), float("inf"), model_epoch_performance
            except PPLExceedingException as e:
                LOG.info(f'PPLExceedingException :: {e}')
                self.handle_exception(MODEL_CHECKPOINT_PATH, model.identifier)
                return float("inf"), float("inf"), model_epoch_performance
            except RuntimeError as e:
                LOG.info(f'Unknown exception for {self.model_identifier} :: {e}')
                self.handle_exception(MODEL_CHECKPOINT_PATH, model.identifier)
                return float("inf"), float("inf"), model_epoch_performance
        else:
            LOG.info(f'An existing model for {self.model_identifier} was found, restoring instead.')

        NASController.stop_training_time_for_architecture(model.identifier)
        LOG.info(f'Loading best model for {self.model_identifier} for final evaluation.')
        if os.path.exists(f'{OUTPUT}/models/{self.model_identifier}{opt_override_txt}_ptb.tar'):
            ModelPersistence.load_model(f'{self.model_identifier}{opt_override_txt}_ptb', model)
        else:
            LOG.info(f'Tried to load best model for {self.model_identifier} but file does not exist.')

        test_loss = self.evaluate(model, test_data, criterion)
        try:
            if test_loss < 10.0e+3 and math.exp(test_loss) < 10.0e+3:
                LOG.info('=' * 89)
                LOG.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                    test_loss, math.exp(test_loss)))
                LOG.info('=' * 89)
            else:
                LOG.info(f'End of training for {self.model_identifier}; Loss value too high, returning 10.0e+3.')
                self.remove_checkpoint(MODEL_CHECKPOINT_PATH)
                return float("inf"), float("inf"), model_epoch_performance
        except OverflowError:
            LOG.info(
                f'End of training for {self.model_identifier}; OverflowError when evaluating performance, returning 10.0e+3.')
            self.remove_checkpoint(MODEL_CHECKPOINT_PATH)
            return float("inf"), float("inf"), model_epoch_performance

        self.remove_checkpoint(MODEL_CHECKPOINT_PATH)

        return test_loss, math.exp(test_loss), model_epoch_performance

    def handle_exception(self, MODEL_CHECKPOINT_PATH, identifier):
        self.remove_checkpoint(MODEL_CHECKPOINT_PATH)
        NASController.stop_training_time_for_architecture(identifier)

    def remove_checkpoint(self, PATH):
        if os.path.exists(PATH):
            os.remove(PATH)

    def train(self, model, epoch, lr, train_data, criterion, optimizer, parent_batch_performance):

        model.train()
        total_loss = 0
        start_time = time.time()
        if EnvironmentConfig.get_config('simulate_results'):
            time.sleep(RandomGenerator.uniform(1, 10))
            return

        hidden = model.init_hidden(self.batch_size)
        diff_count = 0
        previous_loss = 0
        # train_data size(batchcnt, bsz)
        model_batch_performance = {}

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = self.get_batch(train_data, i, DeviceController.get_device())
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.data

            if batch % self.interval == 0 and batch > 0:
                cur_loss = total_loss / self.interval
                elapsed = time.time() - start_time
                if math.isnan(cur_loss):
                    LOG.info(f'Loss nan for model, returning. :: {total_loss}')
                    if self.epoch_loss_nan_count >= 3:
                        raise PPLExceedingException('Training loss nan for three or more consecutive epochs.')

                    self.epoch_loss_nan_count += 1
                    return

                try:
                    LOG.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.3f} | ms/batch {:5.2f} | '
                             'loss {:5.3f} | ppl {:8.3f}'.format(
                        epoch, batch, len(train_data) // args.bptt, lr,
                                      elapsed * 1000 / self.interval, cur_loss, math.exp(cur_loss)))
                except OverflowError:
                    raise PPLExceedingException('Training loss exceeding threshold.')

                model_batch_performance[batch] = {
                    'elapsed': elapsed,
                    'loss': cur_loss
                }

                if batch in parent_batch_performance.keys():
                    parent_elapsed = parent_batch_performance[batch].get('elapsed', 10.0e+9)
                    parent_loss = parent_batch_performance[batch].get('loss', 10.0e+9)

                    performances = []
                    if elapsed > parent_elapsed:
                        performances.append(f'Elapsed diff: {elapsed - parent_elapsed}')
                    if loss > parent_loss:
                        performances.append(f'Loss diff: {loss - parent_loss}')
                    if len(performances) > 0:
                        LOG.info(f'Model performance is decreasing, {". ".join(performances)}.')

                total_loss = 0
                start_time = time.time()
                if previous_loss != 0:
                    diff = cur_loss - previous_loss
                    if diff > 0:
                        diff_count += 1
                    else:
                        diff_count = 0

                    if diff_count >= 3:
                        LOG.info(f'Model is not improving, exit training.')
                        return

                if cur_loss > EnvironmentConfig.get_config('ptb_ppl_upper_threshold'):
                    LOG.info(f'PPL greater than PPL upper threshold. Increasing diff_count.')
                    diff_count += 1

                previous_loss = cur_loss

    def evaluate(self, model, data_source, criterion):
        if EnvironmentConfig.get_config('simulate_results'):
            time.sleep(RandomGenerator.uniform(1, 10))
            return RandomGenerator.uniform(0, 50)

        # Turn on evaluation mode which disables dropout.
        with torch.no_grad():
            model.eval()
            total_loss = 0
            hidden = model.init_hidden(self.eval_batch_size)  # hidden size(nlayers, bsz, hdsize)
            for i in range(0, data_source.size(0) - 1, args.bptt):  # iterate over every timestep
                data, targets = self.get_batch(data_source, i, DeviceController.get_device())
                output, hidden = model(data, hidden)
                # model input and output
                # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
                # output size(bptt*bsz, ntoken)
                total_loss += len(data) * criterion(output, targets).data
                hidden = self.repackage_hidden(hidden)
            return total_loss / len(data_source)

    def export_onnx(self, model, path, batch_size, seq_len):
        LOG.info('The model is also exported in ONNX format at {}'.
                 format(os.path.realpath(args.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to('cpu')
        hidden = model.init_hidden(batch_size)
        # torch.onnx.export(model, (dummy_input, hidden), path)

    def repackage_hidden(self, h):
        # detach
        if self.model_identifier == 'LSTM':
            return repackage_hidden(h)

        packaged = []
        for i in h:
            packaged.append(tuple(v.clone().detach() for v in i))
        return packaged

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(self, source, i, device):
        # source: size(total_len//bsz, bsz)
        seq_len = min(args.bptt, len(source) - 1 - i)
        # data = torch.tensor(source[i:i+seq_len]) # size(bptt, bsz)
        data = source[i:i + seq_len].clone().detach()
        target = source[i + 1:i + 1 + seq_len].clone().detach().view(-1)
        # target = torch.tensor(source[i+1:i+1+seq_len].view(-1)) # size(bptt * bsz)
        return data.to(device), target.to(device)

    def evaluate_model(self, model):
        _, _, test_data = PTBDataLoader.get_data()
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model.to(DeviceController.get_device())
            criterion.to(DeviceController.get_device())
        test_loss = self.evaluate(model, test_data, criterion)
        try:
            LOG.info('=' * 89)
            LOG.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
            LOG.info('=' * 89)
        except OverflowError:
            return 10.0e+10, 10.0e+10

        return test_loss, math.exp(test_loss)

    def evaluate_model_from_file(self):
        model = self.build_model()
        if torch.cuda.is_available():
            model.to(DeviceController.get_device())
        ModelPersistence.load_model(f'{self.model_identifier}_ptb', model)
        loss, ppl = self.evaluate_model(model)
        return loss, ppl, model

    def count_parameters(self, model):
        LOG.info(model)
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        LOG.info(f'\n{table}\nTotal Trainable Params: {total_params}')
        self.model_params = total_params
        return total_params


class PPLExceedingException(Exception):
    pass
