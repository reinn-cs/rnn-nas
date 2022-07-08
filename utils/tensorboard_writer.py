from torch.utils.tensorboard import SummaryWriter

from utils.logger import LOG

LOG = LOG.get_instance().get_logger()

PATH = './output/runs/'

WRITE_TENSOR_BOARD = True


class TensorBoardWriter:
    """
    A convenient class that handles writing to the PyTorch SummaryWriter for viewing model training progress etc.
    """

    __instance = None

    def __init__(self):
        if TensorBoardWriter.__instance is not None:
            raise Exception('Instance already exist.')

        self.writers = {}
        TensorBoardWriter.__instance = self

    def add_writer(self, identifier, writer):
        self.writers[identifier] = writer

    def add_scalar(self, writer_identifier, scalar_identifier, value, epoch):
        if writer_identifier not in self.writers.keys():
            LOG.debug(f'New TensorBoard writer for {writer_identifier}.')
            new_writer = SummaryWriter(PATH + writer_identifier)
            self.writers[writer_identifier] = new_writer

        self.writers[writer_identifier].add_scalar(scalar_identifier, value, epoch)

    def close_existing_writer(self, writer_identifier):
        if writer_identifier in self.writers.keys():
            self.writers[writer_identifier].close()
            self.writers.pop(writer_identifier)

    def get_writer_for_identifier(self, identifier):
        if identifier not in self.writers.keys():
            new_writer = SummaryWriter(PATH + identifier)
            self.writers[identifier] = new_writer
        return self.writers[identifier]

    def close_all_writers(self):
        for writer in self.writers.keys():
            self.writers[writer].close()
        LOG.debug('All Tensorboard writers closed.')

    def has_writer(self, identifier):
        return identifier in self.writers.keys()

    @staticmethod
    def create_new_writer(identifier):
        if not WRITE_TENSOR_BOARD:
            return

        if not TensorBoardWriter.get_instance().has_writer(identifier):
            new_writer = SummaryWriter(PATH + identifier)
            TensorBoardWriter.get_instance().add_writer(identifier, new_writer)

    @staticmethod
    def close_writer(identifier):
        if not WRITE_TENSOR_BOARD:
            return
        TensorBoardWriter.get_instance().close_existing_writer(identifier)

    @staticmethod
    def write_scalar(writer_identifier, scalar_identifier, value, epoch):
        if not WRITE_TENSOR_BOARD:
            return
        TensorBoardWriter.get_instance().add_scalar(writer_identifier, scalar_identifier, value, epoch)

    @staticmethod
    def write_h_params(identifier, lr, bsize, accuracy, loss):
        if not WRITE_TENSOR_BOARD:
            return
        writer = TensorBoardWriter.get_instance().get_writer_for_identifier(identifier)
        writer.add_hparams({'lr': lr, 'bsize': bsize}, {'hparam/accuracy': accuracy, 'hparam/loss': loss})

    @staticmethod
    def write_sine_wave(identifier, fig):
        if not WRITE_TENSOR_BOARD:
            return
        writer = TensorBoardWriter.get_instance().get_writer_for_identifier(identifier)
        writer.add_figure('Sine wave', fig)

    @staticmethod
    def get_instance():
        if TensorBoardWriter.__instance is None:
            TensorBoardWriter()

        return TensorBoardWriter.__instance
