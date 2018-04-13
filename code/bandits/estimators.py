#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 18:24
---------------------------------

"""
import abc


__all__ = ['SampleAverageEstimator', 'ExponentialRecencyWeightedEstimator']


class BaseEstimator(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, reward):
        return None


class SampleAverageEstimator(BaseEstimator):
    def __init__(self, default_value):
        self.initial_value = default_value
        self.value = default_value

        self.n_updates = 0

    def update(self, reward):
        if self.n_updates == 0:
            self.value = reward
        else:
            self.value += (reward - self.value) / self.n_updates

        self.n_updates += 1
        return None


class ExponentialRecencyWeightedEstimator(BaseEstimator):
    def __init__(self, step_size, initial_value):
        self.step_size = step_size
        self.value = initial_value

        self.n_updates = 0

    def update(self, reward):
        self.value += self.step_size * (reward - self.value)
        self.n_updates += 1
        return None




