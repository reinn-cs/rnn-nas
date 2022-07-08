import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch.autograd import Variable

from model.architecture import Architecture
from ops.block_state_builder import BlockStateBuilder
from utils.activation_controller import ActivationController


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, model_arch: Architecture, model_identifier, ntoken, ninp, nhid, nlayers, dropout=0.5, nhidlast=620, post_slack=False):
        super(RNNModel, self).__init__()
        torch.set_default_dtype(torch.double)
        self.identifier = model_identifier

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)  # Token2Embeddings

        self.rnns = []
        for layer in range(nlayers):
            input_size = ninp if layer == 0 else nhid
            hidden_size = nhid
            if self.identifier != 'LSTM':
                self.rnns.append(
                    BlockStateBuilder('').build_new_state_from_architecture(model_arch, input_size, hidden_size,
                                                                            output_layer_dimension=nhid if layer != nlayers - 1 else nhidlast, post_slack=post_slack))
            else:
                self.rnns.append(nn.LSTM(input_size, hidden_size))

        if self.identifier == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l in range(nlayers)]

        self.rnns = nn.ModuleList(self.rnns)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)  # Token2Embeddings
        self.decoder = nn.Linear(nhidlast if self.identifier == 'LSTM' else nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhidlast = nhidlast

    def init_weights(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):

        emb = self.encoder(input)
        if self.training:
            emb = self.drop(emb)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.rnns):

            ActivationController.update_layer(i)

            rnn_outputs = []
            if self.identifier == 'LSTM':
                raw_output, (hx, cx) = rnn(raw_output, hidden[i])
            else:
                raw_output, (hx, cx) = rnn(raw_output, model_states=hidden[i], ptb=True)
            new_hidden.append((hx, cx))
            raw_outputs.append(raw_output)

        hidden = new_hidden
        if self.training:
            output = self.drop(raw_output)
        else:
            output = raw_output
        outputs.append(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        if return_h:
            return decoded, hidden, raw_outputs, outputs

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        if self.identifier == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                     Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                    for l in range(self.nlayers)]

        hidden = []
        for i in range(self.nlayers):
            hidden.append([weight.new_zeros(bsz, self.nhid), weight.new_zeros(bsz, self.nhid)])
        return hidden

    def get_parameters(self):
        parameters = []

        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                parameters.append((name, parameter))

        return parameters

    def get_arch_params(self, print_params=True):
        great_total = 0
        for rnn in self.rnns:
            table = PrettyTable(["Modules", "Parameters"])
            for name, parameter in rnn.named_parameters():
                if not parameter.requires_grad: continue
                param = parameter.numel()
                table.add_row([name, param])
                great_total += param
            if print_params:
                print(table)
        return great_total

class InvalidModelConfigurationException(Exception):
    pass
