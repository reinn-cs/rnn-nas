import os

import torch

from utils.logger import LOG

LOG = LOG.get_instance().get_logger()


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
        if not os.path.exists('./output/models'):
            os.makedirs('./output/models')
        torch.save(model.state_dict(), f'./output/models/{name}.tar')

    @staticmethod
    def load_model(name, model):
        if os.path.exists(f'./output/models/{name}.tar'):
            model.load_state_dict(torch.load(f'./output/models/{name}.tar'), strict=False)
        else:
            LOG.info(f'ERROR :: model {name} was not saved.')
