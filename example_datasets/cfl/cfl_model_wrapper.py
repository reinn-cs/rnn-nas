import torch.nn as nn


class CflModelWrapper(nn.Module):
    def __init__(self, architecture, identifier, input_size, hidden_layer_size, output_layer_size, n_layers,
                 block_state_builder):
        super(CflModelWrapper, self).__init__()
        self.identifier = identifier
        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.rnns = []
        self.encoder = nn.Linear(input_size, hidden_layer_size)
        for l in range(n_layers):
            inp_size = hidden_layer_size # input_size if l == 0 else hidden_layer_size
            self.rnns.append(block_state_builder.build_new_state_from_architecture(architecture, inp_size,
                                                                                   hidden_layer_size,
                                                                                   output_layer_dimension=hidden_layer_size))

        self.rnns = nn.ModuleList(self.rnns)
        self.final = nn.Linear(hidden_layer_size, output_layer_size)

    def forward(self, input, training=True):
        layer_outputs = []
        raw_output = self.encoder(input)
        for i, rnn in enumerate(self.rnns):
            if training:
                raw_output = rnn(raw_output, cfl=True)
            else:
                raw_output, layer_gates = rnn(raw_output, cfl=True, capture_gate_outputs=True)
                layer_outputs.append(layer_gates)

        output = self.final(raw_output)
        return output, layer_outputs

    def get_activation_capture_gates(self):
        return self.rnns[0].get_activation_capture_gates()

    def get_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        return total_params

    def clear_outputs(self):
        for rnn in self.rnns:
            rnn.clear_outputs()
