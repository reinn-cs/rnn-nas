import math

from example_datasets.ptb.ptb_trainer import PtbTrainer
from model.architecture import Architecture


class CompareArchitecturePerformances:
    __instance = None

    def __init__(self):
        if CompareArchitecturePerformances.__instance is not None:
            raise Exception('Instance already exist.')

        CompareArchitecturePerformances.__instance = self

    @staticmethod
    def compare_parent_child_ptb(parent_arch: Architecture, child_arch: Architecture):
        """
        Determine the loss and perplexity differences between parent and offspring architectures for the Penn Treebank
        dataset example.

        :param parent_arch:
        :param child_arch:
        :return:
        """

        try:
            with PtbTrainer(parent_arch, parent_arch.identifier, 0) as parent_trainer:
                parent_loss, parent_ppl, parent_model = parent_trainer.evaluate_model_from_file()

            with PtbTrainer(child_arch, child_arch.identifier, 0) as child_trainer:
                child_model = child_trainer.build_model()
                child_model.load_state_dict(parent_model.state_dict(), strict=False)
                child_loss, child_ppl = child_trainer.evaluate_model(child_model)

            if math.isnan(parent_loss) or math.isnan(parent_ppl) or math.isnan(child_loss) or math.isnan(child_ppl):
                return 10.0e+10, 10.0e+10

            loss_difference = child_loss - parent_loss
            ppl_difference = child_ppl - parent_ppl

            return loss_difference, ppl_difference
        except Exception as e:
            print(e)
            return 10.0e+10, 10.0e+10
