import os
import random

import jsonpickle
import numpy as np
import torch

from example_datasets.ptb.ptb_trainer import PtbTrainer
from ops.nas_controller import NASController
from ops.search_delegator import SearchDelegator
from persistence.persistence import Persistence


def make_dir(_directory):
    if not os.path.exists(_directory):
        os.makedirs(_directory)


def create_directories():
    make_dir('./output')
    make_dir('./output/architectures')
    make_dir('./output/generations')
    make_dir('./output/history')
    make_dir('./output/runs')
    make_dir('./output/sine')
    make_dir('./output/transformations')
    make_dir('./output/persistence')

    make_dir('./restore')
    make_dir('./restore/architectures')

    if os.path.exists('./output/kill.txt'):
        os.remove('./output/kill.txt')
        print('Trigger file removed.')


def restore_architecture(key):
    f = open(f'./restore/architectures/{key}.json')
    json_str = f.read()
    architecture = jsonpickle.decode(json_str)
    f.close()
    return architecture


def test_specific(identifier):
    architecture = restore_architecture(identifier)
    architecture.identifier = identifier

    with PtbTrainer(architecture, identifier, 50) as ptb_trainer:
        test_loss, perplexity, performance = ptb_trainer.run({}, force_training=True)
        model_params = ptb_trainer.model_params
        training_time = NASController.get_architecture_performance_value(identifier, 'training_time')

        print('################')
        print(f'{identifier}:: ppl={perplexity}; time={training_time}; params={model_params}')
        return training_time, perplexity, model_params


if __name__ == '__main__':
    create_directories()
    Persistence.get_instance()

    with SearchDelegator() as NAS_Search:
        NAS_Search.search()
