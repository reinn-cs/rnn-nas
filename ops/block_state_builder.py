import copy

from example_datasets.cfl.cfl_model_wrapper import CflModelWrapper
from example_datasets.sine_wave.sine_model_wrapper import SineModelWrapper
from model.architecture import Architecture
from model.block import Block
from model.circular_reference_exception import CircularReferenceException
from model.state import State
from utils.device_controller import DeviceController
from utils.logger import LOG
from utils.slack_post import SlackPost

LOG = LOG.get_instance()

input_blocks = ['x', 'h', 'c']
output_blocks = ['h_next', 'c_next']


class BlockStateBuilder:
    """
    This class builds a PyTorch model (model.state) from an architecture (model.architecture).
    """

    def __init__(self, identifier):
        self.model_identifier = identifier

    @staticmethod
    def build_new_state_from_architecture(architecture: Architecture, input_layer_dimension, hidden_layer_dimensions,
                                          output_layer_dimension=1, set_final_activation=False, post_slack=False) -> State:
        architecture.clear_block_output_dimensions()

        state = State(input_layer_dimension, hidden_layer_dimensions, output_layer_dimension,
                      elman_network=architecture.elman_network)
        architecture.blocks['x'].output_dimension = input_layer_dimension
        architecture.blocks[
            'h'].output_dimension = hidden_layer_dimensions if architecture.elman_network else output_layer_dimension
        architecture.blocks['c'].output_dimension = hidden_layer_dimensions
        architecture.blocks['h_next'].output_dimension = hidden_layer_dimensions
        architecture.blocks['c_next'].output_dimension = hidden_layer_dimensions

        block_add_count = {}
        blocks_to_add = ['h_next', 'c_next', 'x', 'h', 'c']
        while len(blocks_to_add) > 0:
            next = blocks_to_add.pop()

            if next not in block_add_count.keys():
                block_add_count[next] = 0

            block_add_count[next] = block_add_count[next] + 1

            if block_add_count[next] > len(architecture.blocks.keys())*3:
                message = f'Circular reference for {architecture.identifier} on gate {next} with count {block_add_count[next]}.'
                if post_slack:
                    SlackPost.post_failure('Circular reference', message)
                LOG.info(message)
                raise CircularReferenceException(message)

            if type(next) is int:
                continue

            inputs_not_yet_in_model = []
            for inp in architecture.blocks[next].inputs:
                if inp not in state.blocks.keys() and type(inp) is not int:
                    inputs_not_yet_in_model.append(inp)

            if len(inputs_not_yet_in_model) == 0:
                state.add_block(copy.deepcopy(architecture.blocks[next]))
            else:
                blocks_to_add.append(next)
                blocks_to_add = blocks_to_add + inputs_not_yet_in_model

            if len(blocks_to_add) == 0:
                keys = architecture.get_all_block_keys()
                for key in keys:
                    if key not in state.get_all_block_keys():
                        blocks_to_add.append(key)

        keys = architecture.get_all_block_keys()
        for key in keys:
            if key not in state.get_all_block_keys():
                raise Exception(f'Unknown error occurred while adding {key} to state.')
            else:
                state.blocks[key].get_input_dimensions(state)
                state.blocks[key].get_output_dimension(state)

        if set_final_activation:
            state.set_final_activation_layer()
        state.to(DeviceController.get_device())

        return state

    @staticmethod
    def build_sine_model(architecture: Architecture, input_layer_dimension, hidden_layer_dimensions,
                         output_layer_dimension=1, num_layers=1):
        return SineModelWrapper(architecture, architecture.identifier, input_layer_dimension, hidden_layer_dimensions,
                                output_layer_dimension, num_layers, BlockStateBuilder).double().to(DeviceController.get_device())

    @staticmethod
    def build_cfl_model(architecture: Architecture, input_layer_dimension, hidden_layer_dimensions,
                        output_layer_dimension=1, num_layers=1):
        return CflModelWrapper(architecture, architecture.identifier, input_layer_dimension, hidden_layer_dimensions,
                               output_layer_dimension, num_layers, BlockStateBuilder).to(DeviceController.get_device())

    @staticmethod
    def get_basic_architecture(activation='tanh') -> Architecture:
        architecture = Architecture()

        architecture.add_block(Block([], 'x'))
        architecture.add_block(Block([], 'h'))
        architecture.add_block(Block([], 'c'))

        architecture.add_block(Block(['x'], 'x_lin_r', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_r', activation='linear_b'))
        architecture.add_block(Block(['x_lin_r', 'h_lin_r'], 'x_lin_r_add', combination='add'))

        architecture.add_block(Block(['x_lin_r_add'], 'x_lin_r_actv', activation=activation))

        architecture.add_block(Block(['x_lin_r_actv'], 'h_next'))

        architecture.add_block(Block(['c'], 'c_1'))  # This is required to allow for the add_gate transformation
        architecture.add_block(Block(['c_1'], 'c_next'))

        return architecture

    @staticmethod
    def get_gru_architecture() -> Architecture:
        architecture = Architecture()

        architecture.add_block(Block([], 'x'))
        architecture.add_block(Block([], 'h'))
        architecture.add_block(Block([], 'c'))

        architecture.add_block(Block(['x'], 'x_lin_r', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_r', activation='linear_b'))
        architecture.add_block(Block(['x_lin_r', 'h_lin_r'], 'r_c', combination='add'))
        architecture.add_block(Block(['r_c'], 'r', activation='sigmoid'))

        architecture.add_block(Block(['x'], 'x_lin_z', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_z', activation='linear_b'))
        architecture.add_block(Block(['x_lin_z', 'h_lin_z'], 'z_c', combination='add'))
        architecture.add_block(Block(['z_c'], 'z', activation='sigmoid'))

        architecture.add_block(Block(['x'], 'x_lin_u', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_u', activation='linear_b'))
        architecture.add_block(Block(['h_lin_u', 'r'], 'update_c_1', combination='elem_mul'))
        architecture.add_block(Block(['x_lin_u', 'update_c_1'], 'update_c_2', combination='add'))
        architecture.add_block(Block(['update_c_2'], 'n', activation='tanh'))

        architecture.add_block(Block([1, 'z'], 'z_sub', combination='sub'))
        architecture.add_block(Block(['z_sub', 'n'], 'fw_1', combination='elem_mul'))
        architecture.add_block(Block(['h', 'z'], 'fw_2', combination='elem_mul'))
        architecture.add_block(Block(['fw_1', 'fw_2'], 'h_actv', combination='add'))

        architecture.add_block(Block(['h_actv'], 'h_next'))

        architecture.add_block(Block(['c'], 'c_1'))  # This is required to allow for the add_gate transformation
        architecture.add_block(Block(['c_1'], 'c_next'))

        return architecture

    @staticmethod
    def get_lstm_architecture() -> Architecture:
        architecture = Architecture()

        architecture.add_block(Block([], 'x'))
        architecture.add_block(Block([], 'h'))
        architecture.add_block(Block([], 'c'))

        # Input gate
        architecture.add_block(Block(['x'], 'x_lin_i', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_i', activation='linear_b'))
        architecture.add_block(Block(['x_lin_i', 'h_lin_i'], 'i_add', combination='add'))
        architecture.add_block(Block(['i_add'], 'i', activation='sigmoid'))

        # Forget gate
        architecture.add_block(Block(['x'], 'x_lin_f', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_f', activation='linear_b'))
        architecture.add_block(Block(['x_lin_f', 'h_lin_f'], 'f_add', combination='add'))
        architecture.add_block(Block(['f_add'], 'f', activation='sigmoid'))

        # Output gate
        architecture.add_block(Block(['x'], 'x_lin_o', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_o', activation='linear_b'))
        architecture.add_block(Block(['x_lin_o', 'h_lin_o'], 'o_add', combination='add'))
        architecture.add_block(Block(['o_add'], 'o', activation='sigmoid'))

        # Memory gate
        architecture.add_block(Block(['x'], 'x_lin_m', activation='linear_b'))
        architecture.add_block(Block(['h'], 'h_lin_m', activation='linear_b'))
        architecture.add_block(Block(['x_lin_m', 'h_lin_m'], 'm_add', combination='add'))
        architecture.add_block(Block(['m_add'], 'm', activation='tanh'))

        # Memory state
        architecture.add_block(Block(['f', 'c'], 'f_m_c', combination='elem_mul'))
        architecture.add_block(Block(['i', 'm'], 'i_m_m', combination='elem_mul'))
        architecture.add_block(Block(['f_m_c', 'i_m_m'], 'c_add', combination='add'))

        # Hidden state
        architecture.add_block(Block(['c_add'], 'ct', activation='tanh'))
        architecture.add_block(Block(['ct', 'o'], 'h_m', combination='elem_mul'))

        # Outputs
        architecture.add_block(Block(['h_m'], 'h_next'))
        architecture.add_block(Block(['c_add'], 'c_next'))

        return architecture

    @staticmethod
    def build_model_for_cheap_objectives(_architecture: Architecture) -> State:
        model = State(1, 51, 1)
        architecture = copy.deepcopy(_architecture)
        architecture.blocks['x'].output_dimension = 1
        architecture.blocks['h'].output_dimension = 51
        architecture.blocks['c'].output_dimension = 51
        architecture.blocks['h_next'].output_dimension = 1
        architecture.blocks['c_next'].output_dimension = 1

        blocks_to_add = ['h_next', 'c_next', 'x', 'h', 'c']
        while len(blocks_to_add) > 0:
            next = blocks_to_add.pop()

            if type(next) is int:
                continue

            inputs_not_yet_in_model = []
            for inp in architecture.blocks[next].inputs:
                if inp not in model.blocks.keys() and type(inp) is not int:
                    inputs_not_yet_in_model.append(inp)

            if len(inputs_not_yet_in_model) == 0:
                model.add_block(architecture.blocks[next])
            else:
                blocks_to_add.append(next)
                blocks_to_add = blocks_to_add + inputs_not_yet_in_model

        return model


