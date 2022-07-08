from model.architecture import Architecture


class ObjectiveEvaluation:
    __instance = None

    def __init__(self):
        if ObjectiveEvaluation.__instance is not None:
            raise Exception('Instance already exist.')
        ObjectiveEvaluation.__instance = self

    @staticmethod
    def get_instance():
        if ObjectiveEvaluation.__instance is None:
            ObjectiveEvaluation()

        return ObjectiveEvaluation.__instance

    @staticmethod
    def evaluate_cheap_objectives(_architecture: Architecture, identifier):
        """
        Returns the cheap objectives for the provided architecture.

        Cheap objectives are those that are quick to evaluate (and does not require training the model, for example).

        This method currently constructs a model as it would be used for the Penn Treebank dataset and then determine
        architectural attributes such as the number of parameters the model has, how many blocks the architecture contains,
        and how much multiplication operations does the architecture perform.

        :param _architecture:
        :param identifier:
        :return:
        """

        number_of_blocks = len(_architecture.blocks.keys())

        number_of_add = 0
        number_of_sub = 0
        number_of_mul = 0
        for key in _architecture.blocks.keys():
            block = _architecture.blocks[key]
            if len(block.inputs) > 1 or block.combination is not None:
                if block.combination == 'sub':
                    number_of_sub += 1
                elif block.combination == 'elem_mul':
                    number_of_mul += 1
                else:
                    number_of_add += 1

        return -1, number_of_blocks, number_of_mul, identifier
