import torch
import torch.nn as nn

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SentimentModel(nn.Module):
    def __init__(self, architecture, identifier, block_state_builder, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5, lstm_model=False, basic_rnn=False):
        super(SentimentModel, self).__init__()

        self.identifier = identifier
        self.lstm_model = lstm_model
        self.basic_rnn = basic_rnn

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.lstm_model:
            self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        elif self.basic_rnn:
            self.rnn = nn.RNNCell(embedding_dim, hidden_dim)
        else:
            self.rnns = []
            for l in range(no_layers):
                inp_size = embedding_dim if l == 0 else hidden_dim
                self.rnns.append(block_state_builder.build_new_state_from_architecture(architecture, inp_size,
                                                                                       hidden_dim,
                                                                                       output_layer_dimension=hidden_dim))

            self.last_h_lin = nn.Linear(hidden_dim, hidden_dim)
            self.out_actv = nn.Tanh()

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True

        # print(embeds.shape)  #[50, 500, 1000]
        # lstm_out, hidden = self.lstm(embeds, hidden)

        h0, c0 = hidden
        for input_t in embeds.split(1, dim=1):

            input = torch.squeeze(input_t, 1)
            if self.lstm_model:
                (h0, c0) = self.lstm(input, (h0, c0))
            elif self.basic_rnn:
                h0 = self.rnn(input, h0)
            else:
                # input = input + h0
                for i, rnn in enumerate(self.rnns):
                    (h0, c0) = rnn(input, model_states=(h0, c0), sentiment=True)

                h0 = self.last_h_lin(h0)
                h0 = self.out_actv(h0)

        lstm_out = h0.contiguous().view(-1, self.hidden_dim)

        hidden = (h0, c0)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        # h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)

        h0 = torch.zeros((batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

    def get_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        return total_params
