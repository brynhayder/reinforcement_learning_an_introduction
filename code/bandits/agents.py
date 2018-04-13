#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 18:27
---------------------------------

"""
import numpy as np

__all__ = ['ActionValueBanditAgent']


class ActionValueBanditAgent(object):
    def __init__(self, estimators, actor):
        """
        Agent for bandit problems.

        Args:
            estimators np.array of BaseEstimators: The value estimators.
            actor BaseActor: The thing to choose the actions.
            possible_actions np.array: The actions you can take.

        The order of the arguments is essential. Everything is done on the indices.
        """
        self.estimators = estimators
        self.actor = actor

    def was_exploring(self):
        return self.actor.explore

    def update(self, action, reward):
        self.estimators[action].update(reward)
        return None

    def get_estimates(self):
        return np.array([x.value for x in self.estimators])

    def get_optimal_actions(self):
        values = self.get_estimates()
        return np.where(values == max(values))[0]

    def action(self):
        optimal_actions = self.get_optimal_actions()
        return self.actor.action(optimal_actions)
