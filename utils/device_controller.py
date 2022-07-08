import torch

from utils.logger import LOG

LOG = LOG.get_instance().get_logger()


class DeviceController:
    """
    A convenience class that manages the CUDA device availability (depending on the PyTorch installation and hardware).
    """

    __instance = None

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LOG.info(f'Using device {self.device}.')
        DeviceController.__instance = self

    @staticmethod
    def get_instance():
        if DeviceController.__instance is None:
            DeviceController()

        return DeviceController.__instance

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
