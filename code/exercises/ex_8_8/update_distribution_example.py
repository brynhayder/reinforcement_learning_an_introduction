#!/usr/bin/env python
"""
--------------------------------
project: code
created: 12/07/2018 10:34
---------------------------------

"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from generic import policies

"""
refactoring (model has too many responsibilities
make an agent class and inject the model?



"""


class Model:
    def __init__(self, n_states, n_actions_per_state, branching_factor, random_state=None):
        self.n_states = n_states
        self.n_actions_per_state = n_actions_per_state
        self.branching_factor = branching_factor
        self.random_state = random_state or np.random.RandomState(seed=0)

        self.states = np.arange(n_states)
        self.actions = np.arange(n_actions_per_state)
        self.start_state = 0
        self.terminal_state = n_states - 1
        self.p_terminate = 0.1

        self.non_terminal_states = np.array([s for s in self.states if s != self.terminal_state])
        self.current_state = self.start_state

        self._successor_table = dict()
        for a in self.actions:
            self._successor_table[self.terminal_state, a] = []

        self._expected_reward_table = self.random_state.normal(
                loc=0.,
                scale=1.,
                size=(n_states - 1, n_actions_per_state)
        )
        # self._expected_reward_table = {
        #     tup: self.random_state.normal(loc=0., scale=1.)
        #     for tup in itertools.product(self.non_terminal_states, self.actions)
        # }

    def reset(self):
        self.current_state = self.start_state
        return

    def successors(self, state, action):
        if state == self.terminal_state:
            return []

        if (state, action) not in self._successor_table:
            self._successor_table[state, action] = self.random_state.choice(
                    self.states,
                    size=self.branching_factor
            )
        return self._successor_table[state, action]

    def step(self, action):
        ...

    def expected_reward(self, state, action):
        return self._expected_reward_table[state, action]

    def initial_action_values(self, v):
        avs = {
            s: {a: v for a in self.actions} for s in self.states
        }
        avs[self.terminal_state] = {a: 0. for a in self.actions}
        return avs

    def expected_update(self, action_values, state, action, gamma):
        p = (1 - self.p_terminate) / self.branching_factor
        return (
                self.expected_reward(state, action)
                + gamma * p * sum(max(action_values[s].values()) for s in self.successors(state, action))
        )


class UniformUpdateAgent:
    def __init__(self, model, gamma):



def run_task(n_states, n_actions_per_state, branching_factor, n_iters,
             n_updates_per_task, gamma, tolerance, policy_eval_iter):
    args = [
        (
            Model(
                    n_states=n_states,
                    branching_factor=branching_factor,
                    n_actions_per_state=n_actions_per_state,
                    random_state=np.random.RandomState(seed=seed)
            ),
            n_updates_per_task,
            gamma,
            tolerance,
            policy_eval_iter
        )
        for seed in range(n_iters)
    ]
    with ProcessPoolExecutor() as executor:
        values = executor.map(
                uniform_starting_state_value_estimates,
                *zip(*args)
        )
    return np.column_stack(values)


def uniform_starting_state_value_estimates(model: "Model", n_updates: "int", gamma: "float",
                                           tolerance: "float", policy_eval_iter: "int"):
    av = model.initial_action_values(0.)
    starting_state_values = [0.]
    i = 1
    while i < n_updates:
        for state, action in itertools.product(model.non_terminal_states, model.actions):
            av[state][action] = model.expected_update(av, state, action, gamma=gamma)
            if i % policy_eval_iter == 0:
                vals = policy_evaluation(
                        policy=policies.GreedyPolicy(av, cache=True),
                        model=model,
                        tolerance=tolerance,
                        gamma=gamma
                )
                starting_state_values.append(vals[model.start_state])
            if i == n_updates:
                break
            i += 1

    return np.array(starting_state_values)


def policy_evaluation(policy, model, tolerance, gamma, max_updates=10 ** 5):
    """for deterministic policies only"""
    values = np.array(
            [max(policy.action_values[s].values()) if s != model.terminal_state else 0. for s in model.states]
    )
    delta = 0.
    i = 0
    while True:
        old_values = values
        for s in model.non_terminal_states:
            a = policy(s)
            values[s] = (
                    model.expected_reward(s, a)
                    + gamma * ((1 - model.p_terminate) / model.branching_factor)
                    * sum(values[ns] for ns in model.successors(s, a))
            )
            i += 1
            if i == max_updates:
                return values

        delta = max(delta, np.max(np.abs(values - old_values)))
        if delta < tolerance:
            return values


if __name__ == "__main__":
    import matplotlib;

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    import plotting
    import constants as c

    policy_eval_iter = 500
    n_updates_per_task = 2 * 10 ** 3

    # uniform example

    output = run_task(
            n_states=1000,
            n_actions_per_state=2,
            branching_factor=1,
            n_iters=200,
            n_updates_per_task=n_updates_per_task,
            tolerance=0.01,
            gamma=0.9,
            policy_eval_iter=policy_eval_iter
    )

    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)

        x = np.arange(output.shape[0]) * policy_eval_iter
        ax.plot(x, np.mean(output, axis=1), color='k', lw=3)
        ax.grid(alpha=0.1)

        print('ready')
        plt.show()
