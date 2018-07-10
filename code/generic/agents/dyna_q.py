#!/usr/bin/env python
"""
--------------------------------
project: code
created: 10/07/2018 15:50
---------------------------------

"""
from collections import defaultdict

import numpy as np

from generic import utils, updates


class DynaQAgent:
    def __init__(self, policy, n_plan_iter, alpha, gamma, random_state=None):
        self.policy = policy
        self.n_plan_iter = n_plan_iter
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state or np.random.RandomState(seed=0)

        self._model_table = defaultdict(dict)

    def set_policy_type(self, policy_class, **params):
        self.policy = policy_class(self.policy.action_values, **params)
        return None

    def choose_action(self, state):
        return self.policy(state)

    def model(self, state, action):
        return self._model_table[state][action]

    def update(self, old_state, action, reward, new_state):
        self.update_action_values(old_state, action, reward, new_state)
        self.update_model(old_state, action, reward, new_state)
        self.plan(n=self.n_plan_iter)
        return None

    def plan(self, n):
        for _ in range(n):
            state = utils.choose_from(list(self._model_table.keys()), self.random_state)
            action = utils.choose_from(list(self._model_table[state].keys()), self.random_state)
            reward, new_state = self.model(state, action)
            self.update_action_values(state, action, reward, new_state)
        return None

    def update_action_values(self, old_state, action, reward, new_state):
        updates.q_learning(
                action_values=self.policy.action_values,
                old_state=old_state,
                action=action,
                reward=reward,
                new_state=new_state,
                alpha=self.alpha,
                gamma=self.gamma
        )
        return None

    def update_model(self, old_state, action, reward, new_state):
        self._model_table[old_state][action] = reward, new_state
        return None


class DynaQPlusAgent(DynaQAgent):
    def __init__(self, policy, n_plan_iter, alpha, gamma, kappa=0.01, random_state=None):
        super().__init__(policy, n_plan_iter, alpha, gamma, random_state)

        self.kappa = kappa
        self._time_last_visited_table = defaultdict(dict)
        self._time_step = 0

    def choose_action(self, state):
        self._time_step += 1
        return super().choose_action(state)

    def time_since_visited(self, state, action):
        try:
            t = self._time_last_visited_table[state][action]
        except KeyError:
            t = -1
        return self._time_step - t

    def update(self, old_state, action, reward, new_state):
        super().update(old_state, action, reward, new_state)
        self._time_last_visited_table[old_state][action] = self._time_step
        return None

    def model(self, state, action):
        """Altered simulation rewards for time since taken action"""
        reward, new_state = super().model(state, action)
        r = reward + self.kappa * np.sqrt(self.time_since_visited(state, action))
        return r, new_state
