#!/usr/bin/env python
"""
--------------------------------
project: code
created: 26/06/2018 17:26
---------------------------------

"""
from collections import defaultdict

import numpy as np

from generic.utils import choose_from


def greedy(state_action_values):
    return max(state_action_values, key=lambda k: state_action_values[k])


# not sure if these policy classes should keep references to the action values. Maybe not...
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


class TimeBiasedPolicy:
    def __init__(self, action_values, kappa):
        self.action_values = action_values

        self.kappa = kappa
        self._time_last_visited_table = defaultdict(dict)
        self._time_step = 0

    def time_since_visited(self, state, action):
        try:
            t = self._time_last_visited_table[state][action]
        except KeyError:
            t = -1
        return self._time_step - t

    def __call__(self, state):
        modified_av = {
            a: v + self.kappa * np.sqrt(self.time_since_visited(state, a))
            for a, v in self.action_values[state].items()
        }
        chosen_action = greedy(modified_av)
        self._time_last_visited_table[state][chosen_action] = self._time_step
        self._time_step += 1
        return chosen_action
