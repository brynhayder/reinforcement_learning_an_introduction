#!/usr/bin/env python
"""
--------------------------------
project: code
created: 02/07/2018 14:00
---------------------------------


We set \gamma = 1 in this example: **which makes the two methods equivalent**

Note: the methods are programmed in a way that is specific to this problem

"""
from collections import namedtuple

import numpy as np

Episode = namedtuple('Episode', ['states', 'reward'])


class RandomWalkEnvironment:
    def __init__(self, starting_state, random_state):
        self.starting_state = starting_state
        self.random_state = random_state

        self.current_state = starting_state
        self.n_states = 2 * starting_state - 1
        self.states = list(range(1, self.n_states + 1))
        self.terminal_states = (self.states[0] - 1, self.states[-1] + 1)

        self._rewards = dict(zip(self.terminal_states, [0, 1]))

    def next(self):
        if self.current_state in self.terminal_states:
            raise StopIteration()
        self.current_state += self.random_state.binomial(1, 0.5) * 2 - 1
        return self.current_state, self.reward(self.current_state)

    def reset(self):
        self.current_state = self.starting_state
        return None

    def reward(self, state):
        return self._rewards[state] if state in self.terminal_states else 0

    def true_state_values(self):
        return {s: s / (self.n_states + 1) for s in self.states}

    def terminated(self):
        return self.current_state in self.terminal_states

    def generate_episode(self, max_steps=10 ** 5):
        self.reset()

        i = 1
        states = [self.current_state]
        while not self.terminated():
            environment.next()
            states.append(self.current_state)
            i += 1
            if i >= max_steps:
                break
        return Episode(
                states=states,
                reward=self.reward(self.current_state)
        )


def initial_values(environment, v=1.):
    vals = {
        s: v for s in environment.states
    }
    vals.update({s: 0. for s in environment.terminal_states})
    return vals


def rms_error(estimate, truth):
    return np.sqrt(np.mean([(estimate[s] - truth[s]) ** 2 for s in truth]))


class TDNPredictor:
    """specific to this problem"""

    def __init__(self, n, alpha, environment):
        self.n = n
        self.alpha = alpha
        self.environment = environment

        self.true_state_values = environment.true_state_values()
        self.state_values = None
        self.reset()

    def set_alpha(self, a):
        self.alpha = a
        return None

    def reset(self):
        self.state_values = initial_values(self.environment, v=0.)
        return None

    def update(self, t, terminal_time, episode):
        target = episode.reward * (t + self.n >= terminal_time)
        if t + self.n < terminal_time:
            target += self.state_values[episode.states[t + self.n]]
        return target - self.state_values[episode.states[t]]

    def process_episode(self, episode):
        terminal_time = len(episode.states) - 1
        for t, s in enumerate(episode.states[:-1]):
            self.state_values[s] += self.alpha * (
                self.update(t, terminal_time, episode)
            )
        return rms_error(self.state_values, self.true_state_values)


class TDErrorPredictor(TDNPredictor):

    def td_error(self, t, terminal_time, episode):
        out = self.state_values[episode.states[t + 1]] - self.state_values[episode.states[t]]
        return out + episode.reward if t + 1 == terminal_time else out

    def update(self, t, terminal_time, episode):
        # if t == terminal_time:
        #     return episode.reward - self.state_values[episode.states[t]]
        n_left = min(terminal_time - t, self.n)
        return sum(self.td_error(t + k, terminal_time, episode) for k in range(n_left))


def dict_average(dicts):
    return {
        s: np.mean([d[s] for d in dicts]) for s in dicts[0]
    }


def test_predictor(predictor, n_runs=1000, n_episodes_per_run=100, alpha_0=0.5):
    alphas = np.linspace(alpha_0, 0, n_episodes_per_run)
    all_errors = list()
    state_vals = list()
    for i in range(n_runs):
        predictor.reset()
        errors = list()
        for a in alphas:
            predictor.set_alpha(a)
            e = environment.generate_episode()
            errors.append(predictor.process_episode(e))
        all_errors.append(np.array(errors))
        state_vals.append(predictor.state_values)
    return dict_average(state_vals), sum(all_errors) / n_runs


def run_pred(tup):
    x, y = tup
    return test_predictor(x, **y)


def run_step_lengths(environment, predictor_cls, config):
    with ProcessPoolExecutor() as executor:
        results = executor.map(
                run_pred,
                [(predictor_cls(n, 0.5, environment), config) for n in range(1, environment.n_states)]
        )

    vals, errors = zip(*results)
    return vals, [e.T for e in errors]


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from pprint import pprint

    import matplotlib;

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    config = {
        'n_runs': 100,
        'n_episodes_per_run': 100,
        'alpha_0': 0.5
    }

    random_state = np.random.RandomState(seed=3)
    environment = RandomWalkEnvironment(starting_state=3, random_state=random_state)
    tdn_vals, tdn_errors = run_step_lengths(environment, TDNPredictor, config=config)

    random_state = np.random.RandomState(seed=3)
    environment = RandomWalkEnvironment(starting_state=3, random_state=random_state)
    tderror_vals, tderror_errors = run_step_lengths(environment, TDErrorPredictor, config=config)

    for i, d in enumerate(tdn_vals):
        print(i)
        pprint(
                {s: d[s] - tderror_vals[i][s] for s in d}
        )
    pprint(tdn_vals)
    pprint(tderror_vals)

    for i, er in enumerate(tdn_errors, start=1):
        plt.plot(er, label=f"TDN: n={i}", color=f"C{i}")

    for i, er in enumerate(tderror_errors, start=1):
        plt.plot(er, label=f"TD Error: n={i}", color=f"C{i}", ls="--")

    plt.legend(ncol=2)
    plt.show()
