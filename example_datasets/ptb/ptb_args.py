import argparse

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./example_datasets/ptb/data',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--emsize', type=int, default=650)
parser.add_argument('--nhid', type=int, default=650)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--nhidlast', type=int, default=650,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--save', type=str, default='./output/model_test.pt')
# parser.add_argument('--opt', type=str, default='Adam',
parser.add_argument('--opt', type=str, default='SGD',
                    help='SGD, Adam, RMSprop, Momentum')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
args, _ = parser.parse_known_args()
