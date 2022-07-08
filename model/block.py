import torch

from model.circular_reference_exception import CircularReferenceException
from utils.activation_controller import ActivationController


class Block(object):
    """
    The block encoding scheme developed for the study.
    """

    def __init__(self, inputs, identifier, input_dimension=None, output_dimension=None, activation=None,
                 combination=None, immutable=False):
        self.inputs = inputs
        self.identifier = identifier
        self.activation = []
        self.combination = combination
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.chains = []
        self.output = None
        self.transformations = []
        self.immutable = immutable
        if activation is not None:
            self.activation.append(activation)

    def set_activation_layer(self, layer):
        if len(self.activation) == 0:
            raise Exception('')
        if len(self.activation) >= 2:
            self.activation[1] = layer
        else:
            self.activation.append(layer)

    def get_output(self, state):
        if self.identifier in state.block_outputs.keys():
            return state.block_outputs[self.identifier]

        if len(self.inputs) == 1:
            inp_key = self.inputs[0]
            if inp_key in state.block_outputs.keys():
                inp = state.block_outputs[inp_key]
            else:
                inp = state.blocks[inp_key].get_output(state)

            try:
                self.output = self.activate(inp)
            except Exception as e:
                print(
                    f'Failed for block {self.identifier}; self.input = {self.input_dimension}; x = {state.blocks[inp_key].output_dimension}')
                raise e

        elif len(self.inputs) == 2:
            inp_key_1 = self.inputs[0]
            inp_key_2 = self.inputs[1]

            if type(inp_key_1) is int:
                inp_1 = inp_key_1
            elif inp_key_1 in state.block_outputs.keys():
                inp_1 = state.block_outputs[inp_key_1]
            else:
                inp_1 = state.blocks[inp_key_1].get_output(state)

            if inp_key_2 in state.block_outputs.keys():
                inp_2 = state.block_outputs[inp_key_2]
            else:
                inp_2 = state.blocks[inp_key_2].get_output(state)

            self.output = self.combine(inp_1, inp_2)
            try:
                self.output = self.activate(self.output)
            except Exception as e:
                print(f'Failed for block {self.identifier}; self.input = {self.input_dimension}; x = ')
                raise e

        state.block_outputs[self.identifier] = self.output
        return self.output

    def validate(self, state, encountered):
        if self.identifier in ['x', 'h', 'c']:
            return

        if self.identifier not in encountered.keys():
            encountered[self.identifier] = 0

        encountered[self.identifier] = encountered[self.identifier] + 1

        if encountered[self.identifier] > len(state.blocks.keys()) * 3:
            message = f'Circular reference detected in architecture {state.identifier} for block {self.identifier}'
            print(message)
            raise CircularReferenceException(message)

        for i in self.inputs:
            if type(i) is not int:
                state.blocks[i].validate(state, encountered)

    def activate(self, x):
        if len(self.activation) == 2 and self.activation[0] != 'None':
            activated_out = self.activation[1](x)
            ActivationController.write_activation(self.identifier, self.activation[0], activated_out)
            return activated_out
        return x

    def combine(self, x, y):
        if type(x) is not int and x.shape != y.shape:
            if (len(list(y.shape)) > 1) and y.size(1) < 51 and x.size(0) != y.size(1):
                try:
                    if list(x.shape) == [2] and list(y.shape) == [1, 3]:
                        x = x.unsqueeze(dim=0)
                        x = torch.cat((x, torch.zeros(1, 1)), 1)
                    elif y.shape > x.shape:
                        if list(x.shape) == [1, 2]:
                            x = torch.cat((x, torch.zeros(1, 1)), 1)
                        else:
                            x = torch.cat((x, torch.zeros(1)), 0)
                    elif list(x.shape) == [1, 3] and list(y.shape) == [1, 2]:
                        y = torch.cat((y, torch.zeros(1, 1)), 1)
                except Exception as e:
                    print(f'Exception while matching tensors {x} and {y} for {self.identifier} = {e}.')
                    raise e
            elif False and y.shape == torch.Size([1, 20, 600]) and x.shape == torch.Size([20, 400]):
                x = x.unsqueeze(dim=0)
                tns_zeros = torch.zeros(y.shape[0], y.shape[1], (y.shape[2] - x.shape[2])).to(x.device)
                x = torch.cat((x, tns_zeros), 2)
            elif y.shape == torch.Size([64, 100]) and x.shape == torch.Size([64, 256]):
                tns_zeros = torch.zeros(64, 156).to(x.device)
                y = torch.cat((y, tns_zeros), 1)
            elif x.shape == torch.Size([64, 100]) and y.shape == torch.Size([64, 256]):
                tns_zeros = torch.zeros(64, 156).to(x.device)
                x = torch.cat((x, tns_zeros), 1)
            else:
                x_shape = x.shape
                y_shape = y.shape

                if x_shape[0] == y_shape[0]:
                    if x_shape[1] > y_shape[1]:
                        tns_zeros = torch.zeros(x_shape[0], (x_shape[1] - y_shape[1])).to(x.device)
                        y = torch.cat((y, tns_zeros), 1)
                    elif y_shape[1] > x_shape[1]:
                        tns_zeros = torch.zeros(x_shape[0], (y_shape[1] - x_shape[1])).to(x.device)
                        x = torch.cat((x, tns_zeros), 1)

        if self.combination == 'elem_mul':
            return x * y

        if self.combination == 'sub':
            return x - y

        return x + y

    def get_input_dimensions(self, state):
        if len(self.inputs) > 0 and self.input_dimension is None:
            key = self.inputs[0]
            if type(self.inputs[0]) is int:
                key = self.inputs[1]
            self.input_dimension = state.blocks[key].output_dimension
        return self.input_dimension

    def build_block_chain(self, state):
        chain = [self.identifier]
        for i in self.inputs:
            if type(i) is int:
                chain.append(f'{i}')
            else:
                chain = chain + state.blocks[i].build_block_chain(state)
        self.chains = chain
        return self.chains

    def get_str(self):
        actv = None
        if len(self.activation) > 0:
            actv = self.activation[0]

        inps = [str(x) for x in self.inputs]

        return f'[{",".join(inps)}]->{self.combination if self.combination is not None else "_"}->({actv})'

    def get_output_dimension(self, state):

        if self.identifier in ['x', 'h', 'c', 'h_next', 'c_next']:
            return self.output_dimension

        if len(self.activation) > 0 and self.activation[0] in ['linear', 'linear_b']:
            self.output_dimension = state.hidden_layer_size
            return self.output_dimension

        inp_1 = self.inputs[0]
        inp_1_dim = 0 if type(inp_1) is int else state.blocks[inp_1].get_output_dimension(state)
        if len(self.inputs) > 1:
            inp_2 = self.inputs[1]
            inp_2_dim = 0 if type(inp_2) is int else state.blocks[inp_2].get_output_dimension(state)
            self.output_dimension = max(inp_1_dim, inp_2_dim)
        else:
            self.output_dimension = inp_1_dim

        return self.output_dimension

    def __repr__(self):
        return f'({self.__str__()})'

    def __str__(self):
        actv = None
        if len(self.activation) > 0:
            actv = self.activation[0]

        return f'Identifier = {self.identifier}, inputs = {self.inputs}, activation = {actv}, combination = {self.combination}.'

    def get_op(self):
        if self.combination:
            return self.combination

        if len(self.activation) > 0:
            return self.activation[0]

        return self.identifier
