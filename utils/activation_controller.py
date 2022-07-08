from utils.logger import LOG

LOG = LOG.get_instance().get_logger()


class ActivationController:
    """
    A convenience class that collects block activations, which can then be used for visualising saturation.
    Unfortunately, this didn't present any significant value for the Penn Treebank dataset due to the large vocabulary.
    """

    __instance = None

    def __init__(self):
        self.activations = {}
        self.ins = 0
        self.layer = 0
        self.inpt = 0
        self.cfl = False
        self.training = True
        ActivationController.__instance = self

    def _write_activation(self, block_identifier, activation_function, activated_output):
        if self.training:
            return

        if self.ins not in self.activations.keys():
            self.activations[self.ins] = {}

        if block_identifier not in self.activations[self.ins].keys():
            self.activations[self.ins][block_identifier] = {
                'activation_function': activation_function,
                'values': {},
                'min': 0 if activation_function == 'sigm' else -1
            }

        if self.layer not in self.activations[self.ins][block_identifier]['values'].keys():
            self.activations[self.ins][block_identifier]['values'][self.layer] = {}

        if self.cfl:
            self.activations[self.ins][block_identifier]['values'][self.layer].append(
                activated_output.data.cpu().numpy().squeeze().tolist())
        else:
            self.activations[self.ins][block_identifier]['values'][self.layer][self.inpt] = activated_output[
                0].tolist()  # .append(activated_output[0][i])
            # for i in range(activated_output.shape[1]):
            # value = torch.sum(activated_output[0][i]).item() / max(activated_output.shape)
            # self.activations[self.ins][block_identifier]['values'][self.layer] = activated_output[0].tolist() #.append(activated_output[0][i])

    def reset(self):
        self.activations = {}
        self.ins = 0
        self.layer = 0

    @staticmethod
    def write_activation(block_identifier, activation_function, activated_output):
        if activation_function in ['linear', 'linear_b', 'identity']:
            return

        ActivationController.get_instance()._write_activation(block_identifier, activation_function, activated_output)

    @staticmethod
    def get_instance():
        if ActivationController.__instance is None:
            ActivationController()
        return ActivationController.__instance

    @staticmethod
    def update_count(count):
        ActivationController.get_instance().ins = count

    @staticmethod
    def update_layer(layer):
        ActivationController.get_instance().layer = layer

    @staticmethod
    def reset_input():
        ActivationController.get_instance().inpt = 0

    @staticmethod
    def update_input():
        ActivationController.get_instance().inpt = ActivationController.get_instance().inpt + 1

    @staticmethod
    def log_values():
        return ActivationController.get_instance().activations

    @staticmethod
    def set_cfl(cfl):
        ActivationController.get_instance().cfl = cfl

    @staticmethod
    def reset_values():
        ActivationController.get_instance().reset()

    @staticmethod
    def set_training_mode(training):
        ActivationController.get_instance().training = training
