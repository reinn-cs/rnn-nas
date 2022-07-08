import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from config.env_config import EnvironmentConfig
from model.exploding_gradient_exception import ExplodingGradientException
from ops.nas_controller import NASController
from persistence.model_persistence import ModelPersistence
from utils.logger import LOG
from utils.tensorboard_writer import TensorBoardWriter

LOG = LOG.get_instance().get_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Language_Model_Trainer:
    def __init__(self, epochs=None):
        LOG.info('Language_Model_Trainer, new instance.')
        self.build_local_config()
        if epochs is not None:
            self.epochs = epochs
        else:
            self.epochs = self.config['epochs']

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if False:
            print('Trainer deleted')

    def __del__(self):
        if False:
            print('Trainer deleted')

    def train_model(self, model, data_generator, writer_key, parent_performance):
        if EnvironmentConfig.get_config('simulate_results'):
            loss = np.random.uniform(1, 15)
            return loss, {}

        optimizer = self.get_optimizer(model)
        criterion = self.get_loss_function()
        current_loss = 0
        sample_size = int(self.config['sample_size'])
        length_window_start = int(self.config['length_window_start'])
        length_window_end = int(self.config['length_window_end'])
        distrib_type = self.config['distrib_type']

        if True:
            inputs, outputs, s_dst = data_generator.generate_sample(sample_size, length_window_start, length_window_end,
                                                                    distrib_type, False)

            test_inps, test_outputs, test_dst = data_generator.generate_sample(sample_size // 5, length_window_start,
                                                                               length_window_end,
                                                                               distrib_type, False)
        else:
            inputs, outputs, s_dst = data_generator.generate_sample_s(n_range=5, m_range=5, num_samples=500)

            test_inps, test_outputs, test_dst = data_generator.generate_sample_s(n_range=5, m_range=5, num_samples=100,
                                                                                 test=True)

        model_performance = {}
        best_loss = None
        start_time = time.time()
        NASController.start_training_time_for_architecture(writer_key)
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(len(inputs)):
                input_line_tensor = data_generator.line_to_tensor_input(inputs[i])
                target_line_tensor = data_generator.line_to_tensor_output(outputs[i])

                model.zero_grad()
                optimizer.zero_grad()

                output, _ = model(input_line_tensor)
                loss = criterion(output, target_line_tensor)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                epoch_loss += loss.item()
                if math.isnan(loss.item()):
                    raise ExplodingGradientException('Gradient exploding.')
                # TensorBoardWriter.write_h_params(writer_identifier, learning_rate, 1, 0.0, loss.item())

            test_loss = self.test(model, test_inps, test_outputs, data_generator, criterion)
            model_performance[epoch] = {
                "time": time.time() - start_time,
                "test_loss": test_loss
            }

            NASController.add_model_epoch_performance(writer_key, epoch, model_performance)
            LOG.info(f'[{epoch}] - current loss :: {current_loss} - {loss} :: test loss = {test_loss}')
            TensorBoardWriter.write_scalar(writer_key, 'Loss/Language', test_loss, epoch)

            if epoch in parent_performance.keys():
                parent_time = parent_performance[epoch].get('time', -1)  # ["time"]
                parent_loss = parent_performance[epoch].get('test_loss', -1)  # ["test_loss"]

                if model_performance[epoch].get("time", 0) < parent_time:
                    LOG.info(f'{model.identifier} training time is improving.')

                if model_performance[epoch].get("test_loss", 0) < parent_loss:
                    LOG.info(f'{model.identifier} testing loss is improving.')

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                ModelPersistence.save_model(f'{model.identifier}_lang', model)

        NASController.stop_training_time_for_architecture(writer_key)

        # Load best model for test
        ModelPersistence.load_model(f'{model.identifier}_lang', model)
        test_loss = self.test(model, test_inps, test_outputs, data_generator, criterion)
        return test_loss, model_performance

    def test(self, model, inputs, outputs, data_generator, criterion):
        model.eval()

        losses = []
        for i in range(len(inputs)):
            input_line_tensor = data_generator.line_to_tensor_input(inputs[i])
            target_line_tensor = data_generator.line_to_tensor_output(outputs[i])

            with torch.no_grad():
                output, _ = model(input_line_tensor)
                loss = criterion(output, target_line_tensor)

                if math.isnan(loss.item()):
                    raise ExplodingGradientException('Gradient exploding.')
                losses.append(loss.item())

        return sum(losses) / len(losses)

    def test_model(self, model, data_generator, test_input):
        model.eval()

        if not model.eval:
            raise Exception('Model in incorrect state to proceed with training.')

        max_length = int(self.config['length_window_end'])
        epsilon = float(EnvironmentConfig.get_config('epsilon'))
        with torch.no_grad():

            input = data_generator.line_to_tensor_input(test_input)
            output_string = test_input
            output, _ = model(input)

            if len(list(output.shape)) == 1:
                if output.device != 'cpu':
                    predictions = np.int_(output.cpu().numpy() >= epsilon)
                else:
                    predictions = np.int_(output.numpy() >= epsilon)
            else:
                if output.device != 'cpu':
                    predictions = np.int_(output.cpu()[0].numpy() >= epsilon)
                else:
                    predictions = np.int_(output[0].numpy() >= epsilon)

            letter = self.get_letter_from_predictions(predictions, data_generator)
            output_string += ' | ' + letter
            if letter == 'T':
                LOG.info(f'BREAK_2_Prediction = {output_string}')
                return output_string
            new_inp = data_generator.line_to_tensor_input(letter)

            for i in range(max_length):
                output, _ = model(new_inp)
                if output.device != 'cpu':
                    predictions = np.int_(output.cpu()[0].numpy() >= epsilon)
                else:
                    predictions = np.int_(output[0].numpy() >= epsilon)
                letter = self.get_letter_from_predictions(predictions, data_generator)
                if letter == 'T':
                    break
                else:
                    output_string += letter
                new_inp = data_generator.line_to_tensor_input(letter)

            return output_string

    def test_model_compare(self, model, data_generator):
        sample_size = int(self.config['sample_size'])
        length_window_start = int(self.config['length_window_start'])
        length_window_end = int(self.config['length_window_end'])
        distrib_type = self.config['distrib_type']
        criterion = self.get_loss_function()

        if torch.cuda.is_available():
            model.to(device)
            criterion.to(device)

        test_inps, test_outputs, test_dst = data_generator.generate_sample(sample_size // 5, length_window_start,
                                                                           length_window_end,
                                                                           distrib_type, False)
        return self.test(model, test_inps, test_outputs, data_generator, criterion)

    def get_letter_from_predictions(self, predictions, data_generator):
        if predictions.size == 1 or len(predictions) == 1:
            predicted_idx = predictions[0]
        elif predictions[0] == 1 and predictions[1] == 1:
            rn = random.randint(1, 21)
            if rn % 2 == 0:
                predicted_idx = 0
            else:
                predicted_idx = 1
            # LOG.info(f'Choosing random index {predicted_idx}')
        elif predictions[0] == 1:
            predicted_idx = 0
        elif predictions[1] == 1:
            predicted_idx = 1
        else:
            if len(predictions) > 3:
                if predictions[2] == 1:
                    predicted_idx = 2
                else:
                    predicted_idx = 3
            else:
                predicted_idx = 2

        return data_generator.get_symbol_from_index(predicted_idx)

    def get_optimizer(self, model):
        learning_rate = float(self.config['learning_rate'])
        if self.config['optimizer'] == 'RMSprop':
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif self.config['optimizer'] == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    def build_local_config(self):
        self.config = {
            'epochs': int(EnvironmentConfig.get_config('language_model_epochs')),
            'loss_criterion': EnvironmentConfig.get_config('language_model_loss'),
            'optimizer': EnvironmentConfig.get_config('language_model_optimizer'),
            'learning_rate': EnvironmentConfig.get_config('language_model_learning_rate'),
            'sample_size': EnvironmentConfig.get_config('language_model_sample_size'),
            'length_window_start': EnvironmentConfig.get_config('language_model_window_start'),
            'length_window_end': EnvironmentConfig.get_config('language_model_window_end'),
            'distrib_type': EnvironmentConfig.get_config('language_model_distrib_type')
        }
