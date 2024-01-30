import os
import numpy as np
import pickle

class RandomGenerator:
    __instance = None

    def __init__(self):
        if RandomGenerator.__instance is None:
            self.rng = None
            RandomGenerator.__instance = self

    def _set_seed_value(self, seed_value):
        self.rng = np.random.RandomState(seed_value)

    def _save_rng_state(self):
        state = self.rng.get_state()
        p_file = open('./restore/rng_state.pkl', 'wb')
        pickle.dump(state, p_file)
        p_file.close()

    def _restore_rng_state(self, seed):
        if os.path.exists('./restore/rng_state.pkl'):
            file = open('./restore/rng_state.pkl', 'rb')
            pickle_state = pickle.load(file)
            self.rng = np.random.RandomState()
            self.rng.set_state(pickle_state)
            file.close()
            print('restored')
        else:
            self._set_seed_value(seed)
            self._save_rng_state()
            print('new instance')

    def _get_random(self):
        return self.rng.random_sample()

    def _get_random_int(self, low, high=None, size=None):
        result = self.rng.randint(low, high, size)
        self._save_rng_state()
        return result

    def _uniform(self, low=0.0, high=1.0, size=None):
        result = self.rng.uniform(low, high, size)
        self._save_rng_state()
        return result

    def _choice(self, a, size=None, replace=True, p=None):
        result = self.rng.choice(a, size, replace, p)
        self._save_rng_state()
        return result

    def _normal(self, loc=0.0, scale=1.0, size=None):
        result = self.rng.normal(loc, scale, size)
        self._save_rng_state()
        return result

    @staticmethod
    def instance():
        if RandomGenerator.__instance is None:
            RandomGenerator()

        return RandomGenerator.__instance

    @staticmethod
    def setup(seed):
        print(f'Setup with seed: {seed}')
        RandomGenerator.instance()._restore_rng_state(seed)

    @staticmethod
    def get_random():
        return RandomGenerator.instance()._get_random()

    @staticmethod
    def randint(low, high=None, size=None):
        return RandomGenerator.instance()._get_random_int(low, high, size)

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        return RandomGenerator.instance()._uniform(low, high, size)

    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        return RandomGenerator.instance()._choice(a, size, replace, p)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        return RandomGenerator.instance()._normal(loc, scale, size)

    @staticmethod
    def save():
        RandomGenerator.instance()._save_rng_state()
