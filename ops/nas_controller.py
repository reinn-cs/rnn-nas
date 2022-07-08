import os
import time

import jsonpickle
import torch

from model.architecture_objectives import ArchitectureObjectives


class NASController:
    """
    A controller class that keeps track of the NAS state and some model metrics.
    """
    __instance = None

    def __init__(self):
        if NASController.__instance is not None:
            raise Exception('Instance already exist.')

        self.architecture_training_times = {}
        self.architecture_performance = {}
        self.architecture_epoch_performance = {}
        self.generation = 0
        self.setup_model_epoch_performances()

        NASController.__instance = self

    def setup_model_epoch_performances(self):
        if os.path.exists('./restore/model_epoch_performances.json'):
            f = open(f'./restore/model_epoch_performances.json', 'r')
            json_str = f.read()
            self.architecture_epoch_performance = jsonpickle.decode(json_str)
            f.close()

    @staticmethod
    def get_instance():
        if NASController.__instance is None:
            NASController()
            return NASController.__instance
        else:
            return NASController.__instance

    @staticmethod
    def update_generation():
        NASController.get_instance().generation += 1

    @staticmethod
    def start_training_time_for_architecture(identifier: str):
        NASController.get_instance().architecture_training_times[identifier] = {
            'start': time.time(),
            'difference': -1
        }

    @staticmethod
    def stop_training_time_for_architecture(identifier: str):
        start_time = NASController.get_instance().architecture_training_times[identifier]['start']
        end_time = time.time()
        difference = end_time - start_time
        NASController.get_instance().architecture_training_times[identifier]['end'] = end_time
        NASController.get_instance().architecture_training_times[identifier]['difference'] = difference
        NASController.get_instance().set_architecture_performance(identifier, ArchitectureObjectives.TRAINING_TIME,
                                                                  difference)

    @staticmethod
    def get_architecture_training_time(identifier: str):
        return NASController.get_instance().architecture_training_times[identifier]['difference']

    @staticmethod
    def add_model_epoch_performance(identifier, epoch, performance):
        if identifier not in NASController.get_instance().architecture_epoch_performance.keys():
            NASController.get_instance().architecture_epoch_performance[identifier] = {}

        NASController.get_instance().architecture_epoch_performance[identifier][epoch] = performance

    @staticmethod
    def get_model_epoch_performance(identifier, epoch):
        if identifier not in NASController.get_instance().architecture_epoch_performance.keys():
            return {}

        if epoch > -1:
            return NASController.get_instance().architecture_epoch_performance[identifier].get(epoch, {})

        return NASController.get_instance().architecture_epoch_performance[identifier]

    @staticmethod
    def persist_model_epoch_performance():
        model_epoch_performances = {}
        for key in NASController.get_instance().architecture_epoch_performance.keys():
            model_epoch_performances[key] = NASController.get_instance().architecture_epoch_performance[key]

        f = open(f'./restore/model_epoch_performances.json', 'w')
        json_object = jsonpickle.encode(model_epoch_performances)
        f.write(json_object)
        f.close()

    @staticmethod
    def set_architecture_performance(identifier, key, performance_value):
        if identifier not in NASController.get_instance().architecture_performance.keys():
            NASController.get_instance().architecture_performance[identifier] = {}

        NASController.get_instance().architecture_performance[identifier][key.value] = performance_value

    @staticmethod
    def get_architecture_performance_value(identifier, key):
        result = None

        if identifier in NASController.get_instance().architecture_performance.keys():
            result = NASController.get_instance().architecture_performance[identifier].get(key, None)

        return result

    @staticmethod
    def get_current_state():
        if not os.path.exists('./restore/state.pt'):
            return {}
        return torch.load('./restore/state.pt')

    @staticmethod
    def get_results_for_architecture(identifier):
        state = NASController.get_current_state()
        if 'architecture_results' in state.keys():
            return state['architecture_results'].get(identifier, None)
        return None

    @staticmethod
    def add_results_for_architecture(identifier, results):
        state = NASController.get_current_state()
        if 'architecture_results' not in state.keys():
            state['architecture_results'] = {}

        state['architecture_results'][identifier] = results
        torch.save(state, './restore/state.pt')

    @staticmethod
    def set_architectures_currently_evaluating(architectures):
        f = open('./restore/currently_evaluating.json', 'w')
        json_object = jsonpickle.encode(architectures)
        f.write(json_object)
        f.close()

    @staticmethod
    def get_architectures_currently_evaluating():
        if os.path.exists('./restore/currently_evaluating.json'):
            f = open(f'./restore/currently_evaluating.json', 'r')
            json_str = f.read()
            f.close()
            return jsonpickle.decode(json_str)
        return set()
