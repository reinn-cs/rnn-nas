import math

import numpy as np
from torch import Tensor

model_metrics = ["sine_wave_loss", "sine_wave_ma_error", "lang_loss", "lang_accuracy", "ptb_loss", "ptb_ppl",
                 "sentiment_acc", "sentiment_loss"]


class ModelFitness(object):
    def __init__(self, identifier):
        self.identifier = identifier
        self.number_of_parameters = 0
        self.number_of_blocks = 0
        self.number_of_add = 0
        self.number_of_sub = 0
        self.number_of_mul = 0
        self.training_time = 10.0e+3
        self.sine_wave_loss = 0
        self.sine_wave_ma_error = 0
        self.lang_loss = 0
        self.lang_accuracy = 0
        self.previous_fitness = []
        self.ptb_loss = 0
        self.ptb_ppl = 10.0e+3
        self.sentiment_acc = 0
        self.sentiment_loss = 0
        self.sine_wave_performance = []
        self.ptb_performance = {}
        self.lang_performance = {}

    def add_previous_fitness(self, fitness):
        """
        :param fitness: tuple(avg_loss, sine_wave_error, lang_accuracy)
        :return:
        """
        self.previous_fitness.append(fitness)

    def get_fitness_array(self, objectives):
        lst = []
        for objective in objectives:
            if objective == 'lang_accuracy':
                val = getattr(self, objective) * -1
            else:
                val = getattr(self, objective)

            if type(val) is Tensor:
                val = val.cpu().item()

            if math.isnan(val):
                if objective in model_metrics and objective != 'lang_accuracy':
                    val = 10.0e+5
                else:
                    val = 10.0e+5 * -1

            lst.append(val)

        arr = np.array(lst)
        return np.negative(arr)

    @property
    def average_expensive_fitness(self):
        count = 2

        if self.sine_wave_loss == 0:
            count = 1
        if self.lang_loss == 0:
            count = 1

        avg_loss = (self.sine_wave_loss + self.lang_loss) / count
        return avg_loss, self.sine_wave_ma_error, self.lang_accuracy, self.identifier

    def is_sine_fitness_default(self):
        return (self.sine_wave_loss == 0 and
                self.sine_wave_ma_error == 0 and
                self.training_time == 10.0e+3)

    def is_lang_fitness_default(self):
        return (self.lang_loss == 0 and
                self.lang_accuracy == 0 and
                self.training_time == 10.0e+3)

    def is_ptb_fitness_default(self):
        return (self.ptb_loss == 0 and
                self.ptb_ppl == 10.0e+3 and
                self.training_time == 10.0e+3)

    def is_sentiment_default(self):
        return (self.sentiment_acc == 0 and
                self.sentiment_loss == 0 and
                self.training_time == 10.0e+3)

    def get_ptb_fitness(self):
        return self.ptb_loss, self.ptb_ppl, self.training_time

    def __repr__(self):
        return f'\'{self.__str__()}\''

    def __str__(self):
        return f'[{self.identifier}]: {self.number_of_parameters}, {self.number_of_blocks}, {self.ptb_ppl}>'
