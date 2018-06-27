#!/usr/bin/env python
"""
--------------------------------
project: code
created: 26/06/2018 17:26
---------------------------------

"""
import numpy as np

from generic.utils import choose_from


def greedy(state_action_values):
    return max(state_action_values, key=lambda k: state_action_values[k])


class GreedyPolicy:
    def __init__(self, action_values):
        self.action_values = action_values

    def __call__(self, state):
        return greedy(self.action_values[state])


class EpsilonGreedyPolicy:
    def __init__(self, action_values, epsilon, random_state=None):
        self.action_values = action_values
        self.epsilon = epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

    def explore(self):
        return self.random_state.binomial(n=1, p=self.epsilon) == 1

    def __call__(self, state):
        if self.explore():
            return choose_from(list(self.action_values[state].keys()), self.random_state)
        else:
            return greedy(self.action_values[state])

