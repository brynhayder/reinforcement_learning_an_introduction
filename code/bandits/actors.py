#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 18:24
---------------------------------

"""
import abc

import numpy as np


__all__ = ['EpsilonGreedyActor']


class BaseActor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def action(self, optimal_actions):
        pass


# is it worth maintaining more than one random state? one for actions and one for exploration?
class EpsilonGreedyActor(BaseActor):
    def __init__(self, n_actions, epsilon=0.01, random_state=None):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

        self.possible_actions = np.arange(self.n_actions)
        self.explore = None

    def update(self):
        self.explore = self.random_state.binomial(n=1, p=self.epsilon)
        return None

    def explorative_action(self):
        return self.random_state.choice(self.possible_actions, 1)[0]

    def exploitative_action(self, optimal_actions):
        if len(optimal_actions) == 1:
            return optimal_actions[0]
        else:
            return self.random_state.choice(optimal_actions, 1)[0]

    def action(self, optimal_actions):
        self.update()
        if self.explore:
            return self.explorative_action()
        else:
            return self.exploitative_action(optimal_actions)

