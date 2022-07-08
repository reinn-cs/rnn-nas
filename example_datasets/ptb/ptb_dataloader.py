import copy

from example_datasets.ptb import ptb_data
from example_datasets.ptb.ptb_args import args
from utils.device_controller import DeviceController
from utils.logger import LOG

LOG = LOG.get_instance().get_logger()


class PTBDataLoader:
    __instance = None

    def __init__(self, eval_batch_size=10):
        if PTBDataLoader.__instance is not None:
            raise Exception('Instance already exist.')

        self.eval_batch_size = eval_batch_size
        self.corpus = ptb_data.Corpus(args.data)
        self.ntokens = len(self.corpus.dictionary)
        self.train_data = self.batchify(self.corpus.train, args.batch_size)
        self.val_data = self.batchify(self.corpus.valid, self.eval_batch_size)
        self.test_data = self.batchify(self.corpus.test, self.eval_batch_size)

        PTBDataLoader.__instance = self

    @staticmethod
    def get_instance():
        if PTBDataLoader.__instance is None:
            PTBDataLoader()

        return PTBDataLoader.__instance

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.
    def batchify(self, _data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = _data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = _data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(DeviceController.get_device())

    @staticmethod
    def get_ntokens():
        return copy.deepcopy(PTBDataLoader.get_instance().ntokens)

    @staticmethod
    def get_data():
        return copy.deepcopy(PTBDataLoader.get_instance().train_data), \
               copy.deepcopy(PTBDataLoader.get_instance().val_data), \
               copy.deepcopy(PTBDataLoader.get_instance().test_data)

    @staticmethod
    def reload_validation_data(val_batch_size):
        PTBDataLoader.get_instance().val_data = PTBDataLoader.get_instance().batchify(
            PTBDataLoader.get_instance().corpus.valid, val_batch_size)
