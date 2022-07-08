import datetime
import math
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

from config.env_config import EnvironmentConfig
from example_datasets.sine_wave.generate_sine_wave import generate_data
from ops.nas_controller import NASController
from persistence.model_persistence import ModelPersistence
from utils.device_controller import DeviceController
from utils.tensorboard_writer import TensorBoardWriter

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.logger import LOG

LOG = LOG.get_instance().get_logger()


class SineWaveTrainer:
    def __init__(self, epochs=None):
        self.build_local_config()
        if epochs is None:
            self.epochs = self.config['epochs']
        else:
            self.epochs = epochs

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print('Sine_Wave_Trainer deleted.')

    def train_model(self, model, model_identifier, writer_key, generation, parent_epoch_performance, number_of_phases=3,
                    noise_range=0.0):
        if EnvironmentConfig.get_config('simulate_results'):
            test_val = np.random.randint(0, 100)
            if test_val >= 90:
                raise ExplodingGradient('Gradient exploding.')

            return np.random.uniform(0.01, 50.5), np.random.uniform(0.01, 50.5), {}

        optimizer = self.get_optimizer(model)
        criterion = self.get_loss_function()

        # set random seed to 0
        np.random.seed(0)
        torch.manual_seed(0)
        # load data and make training set
        max_phases = 100
        data = generate_data(phases=max_phases, noise_range_p=noise_range)

        training_phases = []
        for i in range(number_of_phases):
            phase = random.randint(0, max_phases - 1)
            while phase in training_phases:
                phase = random.randint(0, max_phases - 1)

            training_phases.append(phase)

        training_data = self.get_sub_data(data, training_phases)

        testing_phases = []
        for i in range(self.config['sine_model_test_phases']):
            phase = random.randint(0, max_phases - 1)
            while phase in training_phases or phase in testing_phases:
                phase = random.randint(0, max_phases - 1)

            testing_phases.append(phase)

        testing_data = self.get_sub_data(data, testing_phases)

        input = torch.from_numpy(training_data[:, :-1]).to(DeviceController.get_device())
        target = torch.from_numpy(training_data[:, 1:]).to(DeviceController.get_device())
        test_input = torch.from_numpy(testing_data[:, :-1]).to(DeviceController.get_device())
        test_target = torch.from_numpy(testing_data[:, 1:]).to(DeviceController.get_device())

        model.double()
        correct = []
        incorrect = []
        # begin to train
        previous_loss = float('inf')
        delta_counter = 0
        start_time = datetime.datetime.now()
        best_val_loss = None
        NASController.start_training_time_for_architecture(model_identifier)
        for i in range(self.epochs):
            LOG.info(f'Sine Wave Epoch: {(i + 1)}')

            def closure():
                optimizer.zero_grad()
                out = model(input)
                loss = criterion(out, target)
                if math.isnan(loss.item()):
                    raise ExplodingGradient('Gradient exploding.')
                loss.backward()
                return loss

            # optimizer.step(closure)
            try:
                optimizer.step(closure)
            except:
                raise ExplodingGradient('Gradient exploding.')
            # begin to predict, no need to track gradient here
            with torch.no_grad():
                future = 1000
                pred = model(test_input, future=future)
                loss = criterion(pred[:, :-future], test_target)
                if pred.device != 'cpu':
                    y = pred.detach().cpu().numpy()
                else:
                    y = pred.detach().numpy()
                if not np.any(np.isnan(y)):
                    s_mean_absolute_error = mean_absolute_error(test_target.cpu(), y[:, :-future])
                else:
                    s_mean_absolute_error = 999
                TensorBoardWriter.write_scalar(writer_key, 'Error/Sine/mean_absolute_error', s_mean_absolute_error, i)
                TensorBoardWriter.write_scalar(writer_key, 'Loss/Sine', loss.item(), i)

                seconds_difference = self.get_seconds_difference(start_time)
                model_epoch_performance = {
                    "loss": loss.item(),
                    "mae": s_mean_absolute_error,
                    "time": seconds_difference
                }
                NASController.add_model_epoch_performance(model_identifier, i, model_epoch_performance)
                if len(parent_epoch_performance) > 0 and i in parent_epoch_performance.keys():
                    _keys = ["loss", "mae", "time"]
                    result = []
                    for k in _keys:
                        if model_epoch_performance[k] < parent_epoch_performance[i][k]:
                            result.append(
                                f'{k} improvement of {parent_epoch_performance[i][k] - model_epoch_performance[k]}')

                    if len(result) > 0:
                        LOG.info(f'{model_identifier} improvement :: {", ".join(result)}.')

                if previous_loss - loss.item() <= 0:
                    delta_counter += 1
                    previous_loss = loss.item()
                    if delta_counter >= 3:
                        raise ExplodingGradient('Model is not learning.')
                else:
                    delta_counter = 0

                if not best_val_loss or loss.item() < best_val_loss:
                    ModelPersistence.save_model(f'{model_identifier}', model)
                    val_loss_str = '{:5.8f}'.format(loss.item())
                    best_val_loss_str = 'None' if not best_val_loss else '{:5.8}'.format(best_val_loss)
                    LOG.info(
                        f'Saved new model for {model_identifier} :: {val_loss_str} vs {best_val_loss_str}')
                    best_val_loss = loss.item()

            # draw the result
            if i == self.epochs - 1 and (generation + 1) % self.config['sine_model_plot_gen_freq'] == 0:
                try:
                    self.plot(i, input, number_of_phases, noise_range, model_identifier, future, y, generation)
                except:
                    print(f'Could not plot for {model_identifier}.')

        NASController.stop_training_time_for_architecture(model_identifier)
        return best_val_loss, s_mean_absolute_error, model_epoch_performance

    def get_optimizer(self, model):
        learning_rate = float(self.config['learning_rate'])
        if self.config['optimizer'] == 'LBFGS':
            return torch.optim.LBFGS(model.parameters(), lr=learning_rate)
        raise Exception('Invalid optimizer specified.')

    def get_loss_function(self):
        if self.config['loss_criterion'] == 'MSELoss':
            return nn.MSELoss()
        elif self.config['loss_criterion'] == 'NLLLoss':
            return nn.NLLLoss()
        elif self.config['loss_criterion'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            lf = self.config['loss_criterion']
            raise Exception(f'Unknown loss function {lf}')

    def get_model_output(self, model, noise_range=0.0):
        # set random seed to 0
        np.random.seed(0)
        torch.manual_seed(0)
        criterion = self.get_loss_function()

        data = generate_data(phases=self.config['sine_model_test_phases'], noise_range_p=noise_range)

        model.eval()
        model.double()
        test_input = torch.from_numpy(data[:, :-1]).to(DeviceController.get_device())
        test_target = torch.from_numpy(data[:, 1:]).to(DeviceController.get_device())
        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            if pred.device != 'cpu':
                y = pred.detach().cpu().numpy()
            else:
                y = pred.detach().numpy()
            loss = criterion(pred[:, :-future], test_target)

            return loss.item(), y

    def build_local_config(self):
        self.config = {
            'epochs': int(EnvironmentConfig.get_config('sine_model_epochs')),
            'loss_criterion': EnvironmentConfig.get_config('sine_model_loss'),
            'optimizer': EnvironmentConfig.get_config('sine_model_optimizer'),
            'learning_rate': EnvironmentConfig.get_config('sine_model_learning_rate'),
            'sample_size': EnvironmentConfig.get_config('sine_model_sample_size'),
            'sine_model_plot_gen_freq': EnvironmentConfig.get_config('sine_model_plot_gen_freq'),
            'sine_model_test_phases': EnvironmentConfig.get_config('sine_model_test_phases'),
            'sine_performance_count_threshold': EnvironmentConfig.get_config('sine_performance_count_threshold'),
            'loss_threshold': EnvironmentConfig.get_config('sine_loss_threshold'),
            'mae_threshold': EnvironmentConfig.get_config('sine_mae_threshold'),
            'seconds_threshold': EnvironmentConfig.get_config('sine_seconds_threshold')
        }

    def plot(self, i, input, number_of_phases, noise_range, model_identifier, future, y, generation):
        plt.figure(figsize=(30, 10))
        plt.title(
            f'Predicted values for {model_identifier} with {number_of_phases} phases at {noise_range} noise range.\n(Dashlines are predicted values{"" if number_of_phases <= 3 else ". First 3 phases shown."})',
            fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':',
                     linewidth=2.0)

        phases_to_plot = number_of_phases if number_of_phases < 3 else 3
        for p in range(phases_to_plot):
            if p == 0:
                col = 'r'
            elif p == 1:
                col = 'g'
            else:
                col = 'b'
            draw(y[p], col)

        now = datetime.datetime.now()
        formatter = '%H_%M_%S'
        format_date = now.strftime(formatter)
        plt.savefig(f'./output/sine/[{generation}]-predicted-{model_identifier}-{i}-{format_date}.png')
        plt.close()

    def plot_model_evaluation(self, model, model_identifier, writer_key, generation, number_of_phases=3,
                              noise_range=0.0):
        # try:
        data = generate_data(phases=self.config['sine_model_test_phases'], noise_range_p=noise_range)
        test_input = torch.from_numpy(data[:, :-1]).to(DeviceController.get_device())
        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            if pred.device != 'cpu':
                y = pred.detach().cpu().numpy()
            else:
                y = pred.detach().numpy()
        self.plot('FINAL', test_input, number_of_phases, noise_range, model_identifier, future, y, generation)

    # except Exception as e:
    #     print(f'Could not plot for {model_identifier} - {e}.')

    def get_sub_data(self, data_inp, phases):
        resulting_data = np.ndarray(shape=(len(phases), 1000), dtype='float64')
        for e, ph in enumerate(phases):
            resulting_data[e] = data_inp[ph].copy()
        return resulting_data

    def compare_with_parent_performance(self, epoch_performance, parent_epoch_performance):
        if len(parent_epoch_performance) < 1:
            return 0

        loss_threshold = self.config['loss_threshold']
        mae_threshold = self.config['mae_threshold']
        seconds_threshold = self.config['seconds_threshold']

        def test_values(v1, v2, th):
            return (v1 - v2) > 0 and (v1 - v2) > th

        worse_performing_count = 0
        for e, v in enumerate(epoch_performance):
            if len(parent_epoch_performance) > e:
                c_loss, c_mae, c_seconds = v
                p_loss, p_mae, p_seconds = parent_epoch_performance[e]

                check = 0
                check += 1 if test_values(c_loss, p_loss, loss_threshold) else 0
                check += 1 if test_values(c_mae, p_mae, mae_threshold) else 0
                check += 1 if test_values(c_seconds, p_seconds, seconds_threshold) else 0

                worse_performing_count += 1 if check == 3 else 0

        return worse_performing_count

    def get_seconds_difference(self, start_time):
        return (datetime.datetime.now() - start_time).total_seconds()


class ExplodingGradient(Exception):
    pass


class InvalidModelConfigurationException(Exception):
    pass
