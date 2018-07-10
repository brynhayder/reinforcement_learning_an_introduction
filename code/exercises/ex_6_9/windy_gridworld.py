#!/usr/bin/env python
"""
--------------------------------
project: code
created: 26/06/2018 13:03
---------------------------------

"""
from collections import namedtuple
import os

import numpy as np

import constants as c
from generic.policies import EpsilonGreedyPolicy, GreedyPolicy
from generic import updates

Episode = namedtuple('Episode', ['states', 'actions', 'rewards'])


def initial_action_values(v, gridworld, possible_actions):
    return {
        s: {a: v if s != gridworld.goal_position else 0. for a in possible_actions} for s in gridworld.possible_states()
    }


class WindyGridWorld:
    def __init__(self, wind_grid, start_position, goal_position):
        self.wind_grid = wind_grid
        self.start_position = start_position
        self.goal_position = goal_position

    def possible_states(self):
        return np.ndindex(*self.wind_grid.shape)

    def run_episode(self, policy, alpha, gamma=1., train_policy=False):
        position = self.start_position
        action = policy(position)

        states = []
        actions = []
        rewards = []

        while position != self.goal_position:
            new_position = self.move(position, action)
            new_action = policy(new_position)
            reward = self.reward(new_position)

            if train_policy:
                updates.sarsa(
                        action_values=policy.action_values,
                        old_state=position,
                        action=action,
                        reward=reward,
                        new_state=new_position,
                        new_action=new_action,
                        alpha=alpha,
                        gamma=gamma
                )

            states.append(position)
            actions.append(action)
            rewards.append(reward)

            position = new_position
            action = new_action

        return Episode(
                states=states,
                actions=actions,
                rewards=rewards
        )

    def reward(self, new_position):
        return 0 if new_position == self.goal_position else -1

    def _clip_to_grid(self, position):
        x, y = position
        return (
            min(max(x, 0), self.wind_grid.shape[0] - 1),
            min(max(y, 0), self.wind_grid.shape[1] - 1)
        )

    def get_wind(self, position):
        return self.wind_grid[position]

    def move(self, from_position, action):
        x, y = from_position
        dx, dy = action
        new_position = (x + dx - self.get_wind(from_position), y + dy)
        # from the diagram in the book, you can see that if you go off the grid then you
        # get your position rounded to the edge of the grid.
        return self._clip_to_grid(new_position)

    def _episode_string(self, episode):
        output = ""
        for i in range(self.wind_grid.shape[0]):
            for j in range(self.wind_grid.shape[1]):
                pos = i, j
                if pos == self.start_position:
                    output += "s"
                elif pos == self.goal_position:
                    output += "f"
                elif pos in episode.states:
                    output += "*"
                else:
                    output += str(self.wind_grid[pos])
                output += " "
            output += "\n"
        return output

    def print_episode(self, episode):
        return print(self._episode_string(episode))


if __name__ == "__main__":
    n_episodes = 500
    alphas = np.linspace(0.5, 0., n_episodes)
    gamma = 1.

    # this reproduces the textbook example.
    # possible_actions = [
    #     (-1, 0),
    #     (0, -1),
    #     (0, 1),
    #     (1, 0),
    # ]

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

    gridworld = WindyGridWorld(
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

    episodes = list()
    total_training_steps = 0
    for i, a in enumerate(alphas):
        e = gridworld.run_episode(
                epsilon_greedy_policy,
                alpha=a,
                gamma=gamma,
                train_policy=True
        )
        episodes.append(e)

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
