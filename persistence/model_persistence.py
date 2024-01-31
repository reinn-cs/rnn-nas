import os

import torch

from config.env_config import EnvironmentConfig
from utils.logger import LOG

LOG = LOG.get_instance().get_logger()


GOOGLE_DRIVE_EXISTS = os.path.exists('/content/drive/My Drive/msc_run')
OUTPUT = f'./output' if not GOOGLE_DRIVE_EXISTS else f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}/output'

class ModelPersistence:
    """
    This class is responsible for saving and loading models.

    It is also used to initialise an offspring model with its parent's parameters.
    """
    __instance = None

    @staticmethod
    def get_instance():
        if ModelPersistence.__instance is None:
            ModelPersistence()
            return ModelPersistence.__instance
        else:
            return ModelPersistence.__instance

    def __init__(self):
        if ModelPersistence.__instance != None:
            raise Exception('Trying to create new instance of existing singleton.')

        ModelPersistence.__instance = self

    @staticmethod
    def save_model(name, model):
        if not os.path.exists(f'{OUTPUT}/models'):
            os.makedirs(f'{OUTPUT}/models')
        torch.save(model.state_dict(), f'{OUTPUT}/models/{name}.tar')

    @staticmethod
    def load_model(name, model):
        if os.path.exists(f'{OUTPUT}/models/{name}.tar'):
            model.load_state_dict(torch.load(f'{OUTPUT}/models/{name}.tar'), strict=False)
        else:
            LOG.info(f'ERROR :: model {name} was not saved.')
