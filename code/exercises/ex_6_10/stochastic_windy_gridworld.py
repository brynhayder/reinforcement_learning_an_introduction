#!/usr/bin/env python
"""
--------------------------------
project: code
created: 27/06/2018 12:53
---------------------------------

"""
import os

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import constants as c
from generic.policy import EpsilonGreedyPolicy, GreedyPolicy

from exercises.ex_6_9.windy_gridworld import WindyGridWorld, initial_action_values


class StochasticWindyGridWorld(WindyGridWorld):
    def __init__(self, wind_grid, start_position, goal_position, random_state=None):
        super().__init__(
                wind_grid=wind_grid,
                start_position=start_position,
                goal_position=goal_position
        )
        self.random_state = random_state or np.random.RandomState(seed=0)

    def get_wind(self, position):
        return super().get_wind(position) + self.random_state.binomial(2, 1/3) - 1


def plot_learning_curve(episodes, ax, **kwargs):
    training_steps = 0
    y = []
    x = []
    for i, episode in enumerate(episodes):
        training_steps += len(episode.rewards)
        y.append(i)
        x.append(training_steps)
    return ax.plot(x, y, **kwargs)


def learning_curve_chart(episodes):
    """make chart like in the book"""
    fig, ax = plt.subplots(1)
    plot_learning_curve(episodes, ax=ax)
    ax.grid(alpha=0.1)

    ax.set_ylabel("Episode")
    ax.set_xlabel("Total Training Steps")
    ax.set_title("Learning Curve", fontsize=14)
    return fig, ax


if __name__ == "__main__":
    n_episodes = 500
    alphas = np.linspace(0.5, 0., n_episodes)
    gamma = 1.

    possible_actions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        # (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    gridworld = StochasticWindyGridWorld(
            wind_grid=np.loadtxt(
                    os.path.join(
                            c.Paths.input,
                            'ex_6_9',
                            'gridworld.csv'
                    ),
                    dtype=int
            ),
            start_position=(3, 0),
            goal_position=(3, 7)
    )

    av = initial_action_values(.5, gridworld, possible_actions)
    epsilon_greedy_policy = EpsilonGreedyPolicy(
            action_values=av,
            epsilon=0.1
    )

    training_episodes = list()
    total_training_steps = 0
    for i, a in enumerate(alphas):
        e = gridworld.run_episode(
                epsilon_greedy_policy,
                alpha=a,
                gamma=gamma,
                train_policy=True
        )
        training_episodes.append(e)

        steps = len(e.rewards)
        total_training_steps += steps
        print(f"Episode {i}: {steps} steps")
    print(f"Total Training Steps: {total_training_steps}")

    greedy_policy = GreedyPolicy(epsilon_greedy_policy.action_values)
    greedy_episode = gridworld.run_episode(
            greedy_policy,
            alpha=0.,
            gamma=gamma,
            train_policy=False
    )

    print("\n\n\n\n\n")
    print(f"Greedy Episode: {len(greedy_episode.rewards)} steps")
    gridworld.print_episode(greedy_episode)

    fig, ax = learning_curve_chart(training_episodes)

    fig.savefig(
            os.path.join(
                    c.Paths.output,
                    'ex_6_10',
                    'learning_curve.eps'
            ),
            dpi=1000,
            format="eps"
    )

