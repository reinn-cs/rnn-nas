from collections import Counter
from random import choice, choices

import numpy as np

from config.env_config import EnvironmentConfig


class ProbabilityDistribution:
    """
    This class can be used to assign higher probabilities to some network transformations.
    This was not used during the study.
    """
    __instance = None

    def __init__(self):
        if ProbabilityDistribution.__instance is not None:
            raise Exception('Instance already exist.')

        self.probability_map = {}
        self.setup_probability_map()
        ProbabilityDistribution.__instance = self

    def setup_probability_map(self):
        keys = ['identity', 'tanh', 'sigmoid', 'relu', 'linear', 'linear_b', 'add', 'elem_mul', 'sub', 'add_unit',
                'add_gate', 'add_connection', 'add_recurrent_connection', 'change_random_activation',
                'change_combination_connection', 'flip_time_delay', 'flip_input', 'flip_output', 'clean_up',
                'remove_unit', 'flip_network_type',
                'remove_gate', 'remove_connection', 'remove_recurrent_connection', 'None', 1, 2, 3, 4, 5]
        for key in keys:
            self.probability_map[key] = EnvironmentConfig.get_config(key)

    def get(self, key):
        if key in self.probability_map.keys():
            return self.probability_map[key]
        return None

    @staticmethod
    def get_probability_for_key(key):
        return ProbabilityDistribution.get_instance().get(key)

    @staticmethod
    def get_instance():
        if ProbabilityDistribution.__instance is None:
            ProbabilityDistribution()
        return ProbabilityDistribution.__instance

    @staticmethod
    def get_distribution(keys, samples=1) -> Counter:
        options = {}
        for key in keys:
            prob = ProbabilityDistribution.get_probability_for_key(key)
            if prob is not None:
                options[key] = prob

        samples = choices(list(options.keys()), list(options.values()), k=samples)
        return Counter(samples)

    @staticmethod
    def get_linear_function():
        options = ['linear', 'linear_b']
        distribution = ProbabilityDistribution.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_activation_function(include_none=False):
        options = EnvironmentConfig.get_config('activation_functions')
        if include_none:
            options.append('None')
        elif 'None' in options:
            options.remove('None')
        distribution = ProbabilityDistribution.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_combination_method(include_none=False):
        options = EnvironmentConfig.get_config('combination_methods')
        if include_none:
            options.append('None')
        distribution = ProbabilityDistribution.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_number_of_transformations():
        return np.random.randint(1, EnvironmentConfig.get_config('override_transformation_count')+1)

    @staticmethod
    def get_uniform_number_of_transformations(max=5):
        options = [i for i in range(1, max + 1)]
        return choice(options)
