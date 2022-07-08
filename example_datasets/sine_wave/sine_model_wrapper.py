import torch
import torch.nn as nn

from utils.device_controller import DeviceController


class SineModelWrapper(nn.Module):
    def __init__(self, architecture, identifier, input_size, hidden_layer_size, output_layer_size, n_layers,
                 block_state_builder):
        super(SineModelWrapper, self).__init__()
        self.identifier = identifier
        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.rnns = []
        for l in range(n_layers):
            inp_size = input_size if l == 0 else hidden_layer_size
            self.rnns.append(block_state_builder.build_new_state_from_architecture(architecture, inp_size,
                                                                                   hidden_layer_size,
                                                                                   output_layer_dimension=hidden_layer_size))
        self.rnns = nn.ModuleList(self.rnns)
        self.final = nn.Linear(hidden_layer_size, output_layer_size)

    def forward(self, input, future=0):

        hidden_states = []
        for _ in range(self.n_layers):
            h_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double).to(DeviceController.get_device())
            c_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.double).to(DeviceController.get_device())
            hidden_states.append((h_t, c_t))

        outputs = []
        for input_t in input.split(1, dim=1):
            for i, rnn in enumerate(self.rnns):
                (h_t, c_t) = hidden_states[i]
                if i == 0:
                    h_n, c_n = rnn(input_t, model_states=(h_t, c_t), sine=True)
                else:
                    h_n, c_n = rnn(h_n, model_states=(h_t, c_t), sine=True)
                hidden_states[i] = (h_n, c_n)

            output = self.final(h_n)
            outputs += [output]

        for i in range(future):
            for i, rnn in enumerate(self.rnns):
                (h_t, c_t) = hidden_states[i]
                if i == 0:
                    h_n, c_n = rnn(output, model_states=(h_t, c_t), sine=True)
                else:
                    h_n, c_n = rnn(h_n, model_states=(h_t, c_t), sine=True)
                hidden_states[i] = (h_n, c_n)

            output = self.final(h_n)
            outputs += [output]

        return torch.cat(outputs, dim=1)

    def get_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        return total_params
