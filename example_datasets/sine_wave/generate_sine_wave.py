import numpy as np
import torch


def generate_data(phases=3, noise_range_p=0.0, save_file=False):
    """
        Generates a SINE wave
    """
    np.random.seed(2)

    T = 20
    L = 1000
    N = 100

    noise = np.random.normal(0, noise_range_p, L)

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64') + noise

    sample = data[:phases]

    if save_file:
        torch.save(sample, open('traindata.pt', 'wb'))

    return sample
