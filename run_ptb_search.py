import os
import random
import sys
import jsonpickle
import numpy as np
import torch

from config.env_config import EnvironmentConfig
from example_datasets.ptb.ptb_trainer import PtbTrainer
from ops.nas_controller import NASController
from ops.search_delegator import SearchDelegator
from persistence.persistence import Persistence

SEED = 111111

from utils.random_generator import RandomGenerator

GOOGLE_DRIVE_EXISTS = os.path.exists('/content/drive/My Drive/msc_run')
RESTORE = f'./restore' if not GOOGLE_DRIVE_EXISTS else f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}/restore'
OUTPUT = f'./output' if not GOOGLE_DRIVE_EXISTS else f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}/output'

def make_dir(_directory):
    if not os.path.exists(_directory):
        os.makedirs(_directory)


def create_directories():
    make_dir(f'{OUTPUT}')
    make_dir(f'{OUTPUT}/architectures')
    make_dir(f'{OUTPUT}/generations')
    make_dir(f'{OUTPUT}/history')
    make_dir(f'{OUTPUT}/runs')
    make_dir(f'{OUTPUT}/sine')
    make_dir(f'{OUTPUT}/transformations')
    make_dir(f'{OUTPUT}/persistence')

    make_dir(RESTORE)
    make_dir(f'{RESTORE}/architectures')

    if os.path.exists(f'{OUTPUT}/kill.txt'):
        os.remove(f'{OUTPUT}/kill.txt')
        print('Trigger file removed.')


def restore_architecture(key):
    f = open(f'{RESTORE}/architectures/{key}.json')
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

    if len(sys.argv) > 1:
        SEED = int(sys.argv[1])

    create_directories()
    Persistence.get_instance()
    RandomGenerator.setup(SEED)
    with SearchDelegator() as NAS_Search:
        NAS_Search.search()
