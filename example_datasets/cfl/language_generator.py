import collections
import json
import os

import numpy as np
import torch
from scipy.special import gamma

START_SYMBOL = 'S'
TERMINATION_SYMBOL = 'T'
epsilon = 0.5  # Epsilon value -- output threshold (during test time)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Language_Generator:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)

        self.available_symbols = self.vocabulary + TERMINATION_SYMBOL
        self.symbol_count = len(self.available_symbols)
        self.symbol_indices = {elt: i for i, elt in enumerate(self.available_symbols)}

        self.extra_symbol = chr(ord(vocabulary[-1]) + 1)  ## a or b (denoted a/b)

    def generate_sample(self, sample_size=1, minv=1, maxv=50, distrib_type='uniform', distrib_display=False):
        input_arr = []
        output_arr = []

        ## domain = [minv, ...., maxv]
        domain = list(range(minv, maxv + 1))

        nums = self.sample_from_a_distrib(domain, sample_size, distrib_type)

        for num in nums:
            i_seq = ''.join(elt for elt in self.vocabulary for _ in range(num))
            o_seq = self.extra_symbol * num  # a or b
            for i in range(1, self.vocabulary_size):
                o_seq += self.vocabulary[i] * ((num - 1) if i == 1 else num)  # b / other letters
            o_seq += 'T'  # termination symbol

            input_arr.append(i_seq)
            output_arr.append(o_seq)

        ## Display the distribution of lengths of the samples
        if distrib_display:
            print('Distribution of the length of the samples: {}'.format(collections.Counter(nums)))

        return input_arr, output_arr, collections.Counter(nums)

    def get_word(self, n_range, m_range):
        n = np.random.randint(1, n_range)
        m = np.random.randint(1, m_range)
        input = f'{"a" * n}{"b" * m}{"c" * n}'
        output = f'{"d" * n}{"b" * (m - 1)}{"c" * n}T'
        return input, output

    def generate_sample_s(self, n_range, m_range, num_samples, test=False):

        if not test and os.path.exists('./output/reg_language.json'):
            with open('./output/reg_language.json') as file:
                json_file = json.load(file)
                return json_file['inputs'], json_file['outputs'], collections.Counter([])

        inputs = []
        outputs = []
        for i in range(num_samples):
            input, output = self.get_word(n_range, m_range)
            inputs.append(input)
            outputs.append(output)

        if not test:
            json_object = {
                'inputs': inputs,
                'outputs': outputs
            }

            with open('./output/reg_language.json', 'w') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)

        return inputs, outputs, collections.Counter([])

    def sample_from_a_distrib(self, domain, sample_size, distrib_name):
        N = len(domain)
        if distrib_name == 'uniform':
            return np.random.choice(a=domain, size=sample_size)

        elif distrib_name == 'u-shaped':
            alpha = 0.25
            beta = 0.25
            return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, N - 1))

        elif distrib_name == 'right-tailed':
            alpha = 1
            beta = 5
            return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, N - 1))

        elif distrib_name == 'left-tailed':
            alpha = 5
            beta = 1
            return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, N - 1))

        else:
            raise Exception(f'Unknown distribution {distrib_name}')

    def get_vocabulary(self):
        return self.vocabulary

    ## Find letter index from all_letters
    def symbol_to_index(self, symbol):
        return self.available_symbols.find(symbol)

    ## Just for demonstration, turn a letter into a <1 x n_letters> tensor
    def symbol_to_tensor(self, symbol):
        tensor = torch.zeros(1, self.symbol_count, device=device)
        tensor[0][self.symbol_to_index(symbol)] = 1
        return tensor

    ## Turn a line into a <line_length x 1 x n_letters>,
    ## or an array of one-hot letter vectors
    def line_to_tensor_input(self, line):
        # tensor = torch.zeros(len(line), self.vocabulary_size)
        tensor = torch.zeros(len(line), self.symbol_count, device=device)
        for li, symbol in enumerate(line):
            if symbol in self.available_symbols:
                tensor[li][self.symbol_to_index(symbol)] = 1
            else:
                print('Error 1')
        return tensor

    def line_to_tensor_output(self, line):
        tensor = torch.zeros(len(line), self.symbol_count, device=device)
        for li, symbol in enumerate(line):
            if symbol in self.available_symbols:
                tensor[li][self.symbol_to_index(symbol)] = 1
            elif symbol == self.extra_symbol:  # a or b
                tensor[li][self.symbol_to_index('a')] = 1
                tensor[li][self.symbol_to_index('b')] = 1
            else:
                print('Error 2')
        return tensor

    # Turn lines into a <line_length x batch_size x n_letters>,
    # or an array of one-hot letter vectors
    def lines_to_tensor_output(self, lines, mode="input"):
        tensor = torch.zeros(max(map(len, lines)), len(lines),
                             self.vocabulary_size if mode == "input" else self.symbol_count, device=device)
        for i, line in enumerate(lines):
            for li, letter in enumerate(line):
                if mode == "output" and letter == self.extra_symbol:
                    tensor[li][i][:-1] = 1
                else:
                    assert letter in self.available_symbols, "Invalid letter " + letter
                    tensor[li][i][self.symbol_indices[letter]] = 1
        return tensor

    def get_symbol_from_tensor(self, tensor):
        if tensor[0].item() >= epsilon and tensor[1].item() >= epsilon:
            return
        else:
            print(str(tensor[0].item()) + '_' + str(tensor[1].item()))

    def get_symbol_from_index(self, index):
        return self.available_symbols[index]

    ## Beta-Binomial density (pdf)
    def beta_binom_density(self, alpha, beta, k, n):
        return 1.0 * gamma(n + 1) * gamma(alpha + k) * gamma(n + beta - k) * gamma(alpha + beta) / (
                    gamma(k + 1) * gamma(n - k + 1) * gamma(alpha + beta + n) * gamma(alpha) * gamma(beta))

    ## Beta-Binomial Distribution
    def beta_bin_distrib(self, alpha, beta, N):
        pdf = np.zeros(N + 1)

        cumulative = 0.0
        for k in range(N + 1):
            prob = self.beta_binom_density(alpha, beta, k, N)
            pdf[k] = prob

        ## Normalize (to fix small precision errors)
        pdf *= (1. / sum(pdf))
        return pdf


if __name__ == '__main__':
    generator = Language_Generator('abc')

    domain = list(range(5, 10 + 1))

    nums = generator.sample_from_a_distrib(domain, 1, 'uniform')
    print(nums)

    sample = generator.generate_sample_s(n_range=5, m_range=5, num_samples=500)
    print(sample)
