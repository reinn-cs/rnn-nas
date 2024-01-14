import numpy as np
import torch
import torch.nn as nn
import time
from utils.logger import LOG

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# function to predict accuracy
def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


class SentimentTrainer:

    def __init__(self):
        self.vocab = None

    def train(self, model, data_loader):

        # loss and optimization functions
        lr = 0.001

        criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        clip = 5
        epochs = 5
        valid_loss_min = np.Inf
        valid_acc_max = 0
        # train for some number of epochs
        epoch_tr_loss, epoch_vl_loss = [], []
        epoch_tr_acc, epoch_vl_acc = [], []

        batch_size = data_loader.batch_size
        train_loader, valid_loader = data_loader.get_data_loaders()

        # moving to gpu
        model.to(device)

        model_performance = {}
        start_time = time.time()

        for epoch in range(epochs):
            train_losses = []
            train_acc = 0.0
            model.train()
            # initialize hidden state
            h = model.init_hidden(batch_size)
            got_nan = False
            for inputs, labels in train_loader:

                inputs, labels = inputs.to(device), labels.to(device)
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                # h0, h1 = h
                h = tuple([each.data for each in h])

                # c = tuple([each.data for each in c])
                # h1 = tuple([each.data for each in h1])

                model.zero_grad()
                output, h = model(inputs, h)

                # calculate the loss and perform backprop
                if not torch.isnan(output).__contains__(True):
                    loss = criterion(output.squeeze(), labels.float())
                    loss.backward()
                    train_losses.append(loss.item())
                    # calculating accuracy
                    accuracy = acc(output, labels)
                    train_acc += accuracy
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                else:
                    got_nan = True

            if got_nan:
                LOG.info(f'Got nan...')

            val_h = model.init_hidden(batch_size)
            val_losses = []
            val_acc = 0.0
            model.eval()
            for inputs, labels in valid_loader:
                # v_h0 = val_h
                val_h = tuple([each.data for each in val_h])

                # v_c0 = val_c
                # v_c0 = tuple([each.data for each in v_c0])

                inputs, labels = inputs.to(device), labels.to(device)

                output, val_h = model(inputs, val_h)
                if not torch.isnan(output).__contains__(True):
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                    accuracy = acc(output, labels)
                else:
                    accuracy = 0

                val_acc += accuracy

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_acc = train_acc / len(train_loader.dataset)
            epoch_val_acc = val_acc / len(valid_loader.dataset)
            epoch_tr_loss.append(epoch_train_loss)
            epoch_vl_loss.append(epoch_val_loss)
            epoch_tr_acc.append(epoch_train_acc)
            epoch_vl_acc.append(epoch_val_acc)
            LOG.debug(f'Epoch {epoch + 1}')
            LOG.debug(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
            LOG.debug(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
            if epoch_val_loss <= valid_loss_min:
                # torch.save(model.state_dict(), 'output/state_dict.pt')
                LOG.debug('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                    epoch_val_loss))
                valid_loss_min = epoch_val_loss
                valid_acc_max = epoch_val_acc
            LOG.debug(25 * '==')

            model_performance[epoch] = {
                "time": time.time() - start_time,
                "test_loss": epoch_val_acc * 100
            }

        # now = datetime.datetime.now()
        # date_format = '%d_%m_%Y_%H_%M_%S'
        # format_date = now.strftime(date_format)
        # torch.save({
        #     'epoch_tr_loss': epoch_tr_loss,
        #     'epoch_vl_loss': epoch_vl_loss,
        #     'epoch_tr_acc': epoch_tr_acc,
        #     'epoch_vl_acc': epoch_vl_acc
        # }, f'output/{model.identifier}-values-{format_date}.pt')

        return valid_acc_max * 100, model_performance
