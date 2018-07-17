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

    def reset(self):
        self.current_state = self.start_state
        return

    def step(self, action):
        if self.random_state.binomial(1, self.p_terminate) == 1:
            return self.terminal_state, self.expected_reward(self.current_state, action), True
        else:
            new_state = self.random_state.choice(self.successors(self.current_state, action))
            result = new_state, self.expected_reward(self.current_state, action), new_state == self.terminal_state
            self.current_state = new_state
            return result

    def successors(self, state, action):
        if state == self.terminal_state:
            return []

        if (state, action) not in self._successor_table:
            self._successor_table[state, action] = self.random_state.choice(
                    self.states,
                    size=self.branching_factor
            )
        return self._successor_table[state, action]

    def expected_reward(self, state, action):
        return self._expected_reward_table[state, action]

    def initial_action_values(self, v):
        avs = {
            s: {a: v for a in self.actions} for s in self.states
        }
        avs[self.terminal_state] = {a: 0. for a in self.actions}
        return avs


class UniformUpdateAgent:
    def __init__(self, model, gamma):
        self.model = model
        self.gamma = gamma

    def starting_state_value_estimates(self, n_updates, tolerance, policy_eval_iter):
        av = self.model.initial_action_values(0.)
        starting_state_values = [0.]
        i = 0
        while i < n_updates:
            for state, action in itertools.product(self.model.non_terminal_states, self.model.actions):
                av[state][action] = expected_update(
                        model=self.model,
                        action_values=av,
                        state=state,
                        action=action,
                        gamma=self.gamma
                )
                if i % policy_eval_iter == 0 and i != 0:
                    vals = policy_evaluation(
                            policy=policies.GreedyPolicy(av, cache=True),
                            model=self.model,
                            tolerance=tolerance,
                            gamma=self.gamma
                    )
                    starting_state_values.append(vals[self.model.start_state])
                i += 1
                if i == n_updates:
                    break

        return np.array(starting_state_values)


class OnPolicyUpdateAgent:
    def __init__(self, model, gamma):
        self.model = model
        self.gamma = gamma

    def starting_state_value_estimates(self, n_updates, tolerance, policy_eval_iter):
        starting_state_values = [0.]

        policy = policies.EpsilonGreedyPolicy(
                action_values=self.model.initial_action_values(0.),
                epsilon=0.1,
                random_state=np.random.RandomState(seed=0)
        )

        self.model.reset()
        i = 0
        while i < n_updates:
            state = self.model.current_state
            action = policy(state)
            _, _, done = self.model.step(action)

            policy.action_values[state][action] = expected_update(
                    model=self.model,
                    action_values=policy.action_values,
                    state=state,
                    action=action,
                    gamma=self.gamma
            )

            if i % policy_eval_iter == 0 and i != 0:
                vals = policy_evaluation(
                        policy=policies.GreedyPolicy(policy.action_values, cache=True),
                        model=self.model,
                        tolerance=tolerance,
                        gamma=self.gamma
                )
                starting_state_values.append(vals[self.model.start_state])

            if done:
                self.model.reset()
            i += 1

        return np.array(starting_state_values)


def run_task(agent_class, n_states, branching_factor, n_actions_per_state, gamma,
             n_iters, n_updates_per_task, tolerance, policy_eval_iter):
    agents = [
        agent_class(
                model=Model(
                        n_states=n_states,
                        branching_factor=branching_factor,
                        n_actions_per_state=n_actions_per_state,
                        random_state=np.random.RandomState(seed=seed),
                ),
                gamma=gamma
        ) for seed in range(n_iters)
    ]

    futures = list()
    with ProcessPoolExecutor() as executor:
        for agent in agents:
            futures.append(
                    executor.submit(
                            agent.starting_state_value_estimates,
                            n_updates_per_task,
                            tolerance,
                            policy_eval_iter
                    )
            )
    return np.column_stack([future.result() for future in futures])


def expected_update(model, action_values, state, action, gamma):
    p = (1 - model.p_terminate) / model.branching_factor
    return (
            model.expected_reward(state, action)
            + gamma * p * sum(max(action_values[s].values()) for s in model.successors(state, action))
    )


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
    import pickle
    import time
    import os

    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    import plotting
    import constants as c
    from exercises import utils as ex_utils

    start = time.time()

    n_states = 1000
    n_updates_per_task = 20 * n_states
    policy_eval_iter = n_updates_per_task / 40
    n_iters = 200
    branching_factor = 1

    uniform_update_output = run_task(
            agent_class=UniformUpdateAgent,
            n_states=n_states,
            n_actions_per_state=2,
            branching_factor=branching_factor,
            n_iters=n_iters,
            n_updates_per_task=n_updates_per_task,
            tolerance=0.01,
            gamma=0.9,
            policy_eval_iter=policy_eval_iter
    )

    on_policy_update_output = run_task(
            agent_class=OnPolicyUpdateAgent,
            n_states=n_states,
            n_actions_per_state=2,
            branching_factor=branching_factor,
            n_iters=n_iters,
            n_updates_per_task=n_updates_per_task,
            tolerance=0.01,
            gamma=0.9,
            policy_eval_iter=policy_eval_iter
    )

    ex_utils.to_pickle(
            uniform_update_output,
            os.path.join(
                c.Paths.output,
                'ex_8_8',
                f'uniform_output_{n_states}_{branching_factor}.pkl'
            )
    )

    ex_utils.to_pickle(
            on_policy_update_output,
            os.path.join(
                c.Paths.output,
                'ex_8_8',
                f'on_policy_output_{n_states}_{branching_factor}.pkl'
            )
    )


    def plot_output(ax, output, color, label):
        x = np.arange(output.shape[0]) * policy_eval_iter
        ax.plot(x, output, alpha=0.05, color=color, lw=1, label=None)
        ax.plot(x, np.mean(output, axis=1), color=color, lw=2, label=label)
        ax.plot(x, np.mean(output, axis=1), color='k', lw=5, label=None, zorder=-1)
        return


    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)
        plot_output(ax, uniform_update_output, color='C3', label="Uniform")
        plot_output(ax, on_policy_update_output, color='C2', label="On-Policy")
        ax.grid(alpha=0.1)
        ax.set_title(
                f"Comparison of update distributions for tasks with {n_states} states and $b=${branching_factor}",
        )
        ax.set_ylabel("Value of initial state")
        ax.set_xlabel("Number of updates")
        ax.legend()

        plotting.savefig(
                fig=fig,
                path=os.path.join(
                        c.Paths.output,
                        'ex_8_8',
                        f'update_distribution_comparison_{n_states}_{branching_factor}.png'
                )
        )
        print('ready')
        print('took', time.time() - start, 'seconds')
        plt.show()
