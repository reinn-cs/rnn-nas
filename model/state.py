import math

import torch
import torch.nn as nn

from model.block import Block
from utils.activation_controller import ActivationController
from utils.device_controller import DeviceController
from utils.logger import LOG

LOG = LOG.get_instance().get_logger()

allowable_activation_functions = {
    'identity': nn.Identity(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}

class State(nn.Module):
    """
    An Architecture class is compiled to a PyTorch model using this class.
    This is done to abstract away from the 'model' when performing network transformations.
    """


    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, elman_network=True):
        super(State, self).__init__()
        self.blocks = {}
        self.layers = nn.ModuleDict()
        self.input_layer_size = input_layer_size
        self.hidden_layers = 1
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.final = None
        self.output_blocks = ['h_next', 'c_next']
        self.input_format = torch.double
        self.block_outputs = {}
        self.transformation_history = []
        self.removed_blocks = []
        self.identifier = None
        self.elman_network = elman_network

    def set_final_activation_layer(self):
        self.final = nn.Linear(self.blocks['h_next'].get_output_dimension(self), self.output_layer_size)

    def add_block(self, block: Block):
        if block.identifier == 'x':
            block.output_dimension = self.input_layer_size
        if block.identifier == 'h':
            block.output_dimension = self.hidden_layer_size if self.elman_network else self.output_layer_size
        if block.identifier == 'c':
            block.output_dimension = self.hidden_layer_size

        if block.identifier not in ['h_next', 'c_next']:
            block.get_output_dimension(self)

        if len(block.activation) == 1 and block.activation[0] != 'None':
            layer_encode = block.activation[0]

            if layer_encode == 'linear':
                layer = nn.Linear(block.get_input_dimensions(self), block.output_dimension, bias=False)
            elif layer_encode == 'linear_b':
                layer = nn.Linear(block.get_input_dimensions(self), block.output_dimension)
            else:
                layer = allowable_activation_functions[layer_encode]
            self.layers[block.identifier] = layer
            block.set_activation_layer(layer)

        self.blocks[block.identifier] = block

    def forward(self, input, sine=False, model_states=None, ptb=False, cfl=False, char_nn=False, sentiment=False,
                capture_gate_outputs=False):

        if sine:
            if model_states is None:
                h_size = self.hidden_layer_size if self.elman_network else self.output_layer_size
                self.blocks['h'].output = torch.zeros(input.size(0), h_size, dtype=self.input_format).to(DeviceController.get_device())
                self.blocks['c'].output = torch.zeros(input.size(0), self.hidden_layer_size,
                                                      dtype=self.input_format).to(
                    DeviceController.get_device())
            else:
                (h_t, c_t) = model_states
                self.blocks['h'].output = h_t
                self.blocks['c'].output = c_t

            self.blocks['x'].output = input

            outputs = {
                'output': []
            }

            self.block_outputs = {}
            for out_key in self.output_blocks:
                outputs[out_key] = self.blocks[out_key].get_output(self)

            return outputs['h_next'], outputs['c_next']
        elif ptb:
            if model_states is None:
                h_t = torch.zeros(1, input.size(1), self.hidden_layer_size).to(DeviceController.get_device())
                c_t = torch.zeros(1, input.size(1), self.hidden_layer_size).to(DeviceController.get_device())
            else:
                (h_t, c_t) = model_states

            self.blocks['h'].output = h_t
            self.blocks['c'].output = c_t
            outputs = {
                'output': []
            }

            ActivationController.reset_input()
            for input_t in torch.unbind(input, dim=0):
                self.block_outputs = {}
                self.blocks['x'].output = input_t

                for out_key in self.output_blocks:
                    outputs[out_key] = self.blocks[out_key].get_output(self)

                self.blocks['h'].output = outputs['h_next']
                self.blocks['c'].output = outputs['c_next']

                outputs['output'].append(outputs['h_next'].clone())
                ActivationController.update_input()

            return torch.stack(outputs['output'], dim=0), (outputs['h_next'], outputs['c_next'])

        elif cfl:
            (hx, cx) = self.init_hidden()
            final_outputs = []
            gates_to_keep = []
            all_gate_outputs = {}
            if capture_gate_outputs:
                for b in self.blocks.keys():
                    if len(self.blocks[b].activation) > 0 and self.blocks[b].activation[0] in ['sigmoid', 'tanh']:
                        gates_to_keep.append(b)

            self.blocks['h'].output = hx
            self.blocks['c'].output = cx
            for ii in range(input.size(0)):
                self.blocks['x'].output = input[ii].unsqueeze(dim=0)
                self.block_outputs = {}
                outputs = {}
                for out_key in self.output_blocks:
                    outputs[out_key] = self.blocks[out_key].get_output(self)

                self.blocks['h'].output = outputs['h_next']
                self.blocks['c'].output = outputs['c_next']
                final_outputs.append(outputs['h_next'])

                if len(gates_to_keep) > 0:
                    for _key in gates_to_keep:
                        if _key not in all_gate_outputs.keys():
                            all_gate_outputs[_key] = []
                        all_gate_outputs[_key].append(self.block_outputs[_key])

            if capture_gate_outputs:
                return torch.cat(final_outputs), all_gate_outputs

            return torch.cat(final_outputs)

        elif char_nn:
            if model_states is None:
                h_t = torch.zeros(input.size(1), self.hidden_layer_size).to(DeviceController.get_device())
                c_t = torch.zeros(input.size(1), self.hidden_layer_size).to(DeviceController.get_device())
            else:
                (h_t, c_t) = model_states

            self.blocks['h'].output = h_t
            self.blocks['c'].output = c_t
            outputs = {
                'output': []
            }

            ActivationController.reset_input()
            for input_t in torch.unbind(input, dim=0):
                self.block_outputs = {}
                self.blocks['x'].output = input_t

                for out_key in self.output_blocks:
                    outputs[out_key] = self.blocks[out_key].get_output(self)

                self.blocks['h'].output = outputs['h_next']
                self.blocks['c'].output = outputs['c_next']
                ActivationController.write_activation('h_next', 'h', outputs['h_next'])
                ActivationController.write_activation('c_next', 'c', outputs['c_next'])

                outputs['output'].append(outputs['h_next'].clone())
                ActivationController.update_input()

            return torch.stack(outputs['output'], dim=0), (outputs['h_next'], outputs['c_next'])
        elif sentiment:
            if model_states is None:
                h_t = torch.zeros(input.size(1), self.hidden_layer_size).to(DeviceController.get_device())
                c_t = torch.zeros(input.size(1), self.hidden_layer_size).to(DeviceController.get_device())
            else:
                (h_t, c_t) = model_states

            self.blocks['h'].output = h_t
            self.blocks['c'].output = c_t
            outputs = {
                'output': []
            }

            ActivationController.reset_input()
            for input_t in torch.unbind(input, dim=0):
                self.block_outputs = {}
                self.blocks['x'].output = input_t

                for out_key in self.output_blocks:
                    outputs[out_key] = self.blocks[out_key].get_output(self)

                self.blocks['h'].output = outputs['h_next']
                self.blocks['c'].output = outputs['c_next']

                outputs['output'].append(outputs['h_next'].clone())
                ActivationController.update_input()

            return torch.stack(outputs['output'], dim=0), (outputs['h_next'], outputs['c_next'])
        else:
            raise Exception('Unknown state.')

    def init_hidden(self):
        self.block_outputs = {}
        return (torch.zeros(self.hidden_layers, self.hidden_layer_size, device=DeviceController.get_device())), (
            torch.zeros(self.hidden_layers, self.hidden_layer_size, device=DeviceController.get_device()))

    def get_all_block_keys(self):
        return list(self.blocks.keys())

    def find_blocks_using_block_with_id(self, id):
        block_keys = set()
        for key in self.blocks.keys():
            block = self.blocks[key]
            if id in block.inputs:
                block_keys.add(block.identifier)
        return list(block_keys)

    def verify_blocks_chain(self):
        checked = 0
        for key in self.blocks.keys():
            block = self.blocks[key]
            encountered = []
            for c in block.build_block_chain(self):
                if c in encountered and c not in ['x', 'h', 'c']:
                    raise Exception(f'Block {key} invalid = {block.build_block_chain(self)}.')
                encountered.append(c)
            checked += len(encountered)
        print(f'State verified with {checked} encounters.')

    def verify_blocks(self):
        outputs = ['h_next', 'c_next']
        for output in outputs:
            inputs = self.blocks[output].inputs
            for input in inputs:
                encountered = []
                if type(input) is not int:
                    self.blocks[input].validate(self, encountered)
                # print(Counter(encountered))

    def update_block_activation(self, block_key, new_activation):

        if len(self.blocks[block_key].activation) > 0:
            if self.blocks[block_key].activation[0] == new_activation and block_key in self.layers.keys():
                print(f'No change in block {block_key} activation {new_activation} // {type(self.layers[block_key])}.')
                return

        self.blocks[block_key].activation = []

        if block_key in self.layers.keys():
            self.layers.pop(block_key)

        block = self.blocks[block_key]
        if new_activation == 'linear':
            layer = nn.Linear(block.get_input_dimensions(self), block.output_dimension, bias=False)
        elif new_activation == 'linear_b':
            layer = nn.Linear(block.get_input_dimensions(self), block.output_dimension)
        elif new_activation == 'tanh':
            layer = nn.Tanh()
            block.output_dimension = block.get_input_dimensions(self)
        elif new_activation == 'sigmoid':
            layer = nn.Sigmoid()
            block.output_dimension = block.get_input_dimensions(self)
        elif new_activation == 'identity':
            layer = nn.Identity()
            block.output_dimension = block.get_input_dimensions(self)
        elif new_activation == 'None' or new_activation is None:
            return
        else:
            raise Exception(f'Unknown activation layer.')

        self.blocks[block_key].activation.append(new_activation)
        self.layers[block_key] = layer
        self.blocks[block_key].set_activation_layer(layer)

    def get_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        return total_params

    def get_block_strings(self):
        block_strings = {}

        for key in self.blocks.keys():
            block = self.blocks[key]
            for inp in block.inputs:
                if type(inp) is not int:
                    if inp not in block_strings.keys():
                        block_strings[inp] = self.blocks[inp].get_str()
            if key not in block_strings.keys():
                block_strings[key] = block.get_str()

        return block_strings

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_layer_size)
        for param in self.parameters():
            torch.nn.init.uniform_(param, -stdv, stdv)

    def clear_outputs(self):
        self.block_outputs = {}
        for block in self.blocks.keys():
            self.blocks[block].output = None

    def update_block_dimensions(self, block_that_changed, new_dim):
        for block_key in self.blocks.keys():
            if block_key != block_that_changed:
                if block_that_changed in self.blocks[block_key].inputs:
                    if self.blocks[block_key].input_dimension != new_dim:
                        self.blocks[block_key].input_dimension = new_dim
                        if len(self.blocks[block_key].activation) > 0 and self.blocks[block_key].activation[0] in [
                            'linear', 'linear_b']:
                            if block_key in self.layers.keys():
                                self.layers.pop(block_key)
                            if self.blocks[block_key].activation[0] == 'linear':
                                layer = nn.Linear(self.blocks[block_key].get_input_dimensions(self),
                                                  self.blocks[block_key].output_dimension, bias=False)
                            else:
                                layer = nn.Linear(self.blocks[block_key].get_input_dimensions(self),
                                                  self.blocks[block_key].output_dimension)

                            self.layers[block_key] = layer
                            self.blocks[block_key].activation[1] = layer
                            LOG.info(f'Updated block {block_key} from state, dimensions changed after transformation.')

    def get_activation_capture_gates(self):
        activation_capture_gates = []
        for b in self.blocks.keys():
            if len(self.blocks[b].activation) > 0 and self.blocks[b].activation[0] in ['sigmoid', 'tanh']:
                activation_capture_gates.append(b)

        return activation_capture_gates
