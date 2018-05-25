#!/usr/bin/env python
"""
--------------------------------
project: code
created: 24/05/2018 11:08
---------------------------------

"""
from types import SimpleNamespace

import numpy as np


class RandomWalkingValueSampler(object):
    def __init__(self, n_steps, n_bandits, loc=0., scale=0.01, random_state=None):
        self.n_steps = n_steps
        self.n_bandits = n_bandits
        self.loc = loc
        self.scale = scale
        self.random_state = random_state or np.random.RandomState(seed=0)

    def get_innovations_starting_with_zero(self):
        innovations = self.random_state.normal(
                loc=self.loc,
                scale=self.scale,
                size=(self.n_steps, self.n_bandits)
        )
        innovations[0, :] = 0
        return innovations

    def sample(self, initial_values):
        return np.atleast_2d(initial_values) + np.cumsum(self.get_innovations_starting_with_zero(), axis=0)

    __call__ = sample


def run_single(agent, samples):
    choices = list()
    explore = list()
    for row in samples:
        choice = agent.action()
        explore.append(agent.was_exploring())
        choices.append(choice)
        reward = row[choice]
        agent.update(choice, reward)
    return SimpleNamespace(
            choices=np.array(choices),
            explore=np.array(explore),
            optimal=np.argmax(samples, axis=1)
    )
