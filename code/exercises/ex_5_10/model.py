#!/usr/bin/env python
"""
--------------------------------
project: code
created: 06/06/2018 15:59
---------------------------------

"""
from collections import namedtuple
import itertools

import numpy as np


Episode = namedtuple('Episode', ['states', 'actions', 'rewards'])


class RaceTrack:
    """Two dimensional racetrack"""

    def __init__(self, track_positions, start_positions, finish_positions, random_state=None):
        self.track_positions = track_positions
        self.start_positions = start_positions
        self.finish_positions = finish_positions
        self.random_state = random_state or np.random.RandomState(seed=0)

        self.car_position = None
        self.car_velocity = None

        self.format_dict = {
            'car'   : 'c',
            'start' : 's',
            'finish': 'f',
            'border': '+',
            'track' : ' ',
            'sep'   : ' '
        }

    def new_race(self):
        self.car_position = self.random_starting_position()
        self.car_velocity = self._stationary()
        self.returns = 0
        return None

    def __enter__(self):
        self.new_race()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def _stationary():
        return 0, 0

    def finished(self):
        return self.car_position in self.finish_positions

    def on_track(self, position):
        return position in self.track_positions

    def random_starting_position(self):
        if len(self.start_positions) == 1:
            return self.start_positions[0]
        return self.random_state.choice(self.start_positions)

    def accelerate(self, dvx, dvy):
        dx, dy = self.car_velocity
        return self.set_velocity(dx + dvx, dy + dvy)

    # This thing has a side effect and a return value...
    def set_velocity(self, dx, dy):
        x, y = self.car_position
        new_position = x + dx, y + dy

        if self.on_track(new_position):
            self.car_position = new_position
            self.car_velocity = (dx, dy)
        else:
            self.car_position = self.random_starting_position()
            self.car_velocity = self._stationary()
        return -1  # reward for the transition

    def __str__(self):
        return self.track_string(self.format_dict)

    def track_string(self, format_dict):
        x, y = zip(*self.track_positions)
        output = ""
        y_rng = range(max(y) + 1)
        for i in range(max(x) + 1):
            for j in y_rng:
                pos = i, j
                if pos == self.car_position:
                    output += format_dict['car']
                elif pos in self.start_positions:
                    output += format_dict['start']
                elif pos in self.finish_positions:
                    output += format_dict['finish']
                elif pos in self.track_positions:
                    output += format_dict['track']
                else:
                    output += format_dict['border']
                output += format_dict['sep']
            output += "\n"
        return output


class Car:
    def __init__(self, policy, speed_limit=5, max_a=1):
        self.policy = policy
        self.speed_limit = speed_limit
        self._velocity_increments = list(
                itertools.product(range(-max_a, max_a + 1), repeat=2)
        )[::-1]

    def possible_actions(self, state):
        x, v = state
        dx, dy = v
        return [
            (ax, ay) for ax, ay in self._velocity_increments
            if dx + ax <= self.speed_limit and dy + ay <= self.speed_limit
        ]

    # When we generate the possible states here we are not completely accurate, in that
    # we generate some unreachable states (i.e. having max velocity after 1 timestep).
    # None of these states are reachable in an episode, so their values won't be updated
    # and the policy in those places is irrelevant.
    def possible_states(self, track):
        # car can only have positive velocity
        velocities = itertools.product(range(self.speed_limit + 1), repeat=2)
        possible_states = [
            tup for tup in itertools.product(
                    track.track_positions,
                    velocities
            )
        ]

        for pos in track.track_positions:
            if pos not in track.start_positions:
                possible_states.remove((pos, (0, 0)))

        return possible_states

    def set_policy(self, policy):
        self.policy = policy
        return None

    def drive(self, current_position, current_velocity):
        return self.policy[current_position, current_velocity]


class Brain:
    @staticmethod
    def initial_action_values(car, racetrack):
        return {
            s: {a: 0. for a in car.possible_actions(s)}
            for s in car.possible_states(racetrack)
        }

    def __init__(self, car, racetrack):
        self.car = car
        self.racetrack = racetrack

        self.action_values = self.initial_action_values(car, racetrack)

    def greedy_policy(self):
        return {
            k: greedy(a) for k, a in self.action_values.items()
        }


def greedy(state_action_values):
    return max(state_action_values, key=lambda k: state_action_values[k])


class EpsilonGreedy:
    def __init__(self, epsilon, random_state):
        self.epsilon = epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

    def explore(self):
        return self.random_state.binomial(1, self.epsilon) == 1

    def __call__(self, state_action_values):
        if len(state_action_values) == 1:
            return next(iter(state_action_values))

        if self.explore():
            return self.random_state.choice(list(state_action_values.keys()))
        else:
            return greedy(state_action_values)


def run_epsiode(car, racetrack):
    """

    Args:
        car (Car):
        racetrack (RaceTrack):

    Returns:
        Episode (Episode): The trajectory.
    """
    with racetrack as race:
        print(race)

        states = [racetrack.car_position]
        actions = list()
        rewards = list()

        while True:
            action = car.drive(
                    current_position=race.car_position,
                    current_velocity=race.car_velocity
            )
            reward = race.accelerate(*action)

            actions.append(action)
            rewards.append(reward)
            states.append(race.car_position)

            print(race)

            if race.finished():
                print("Finished!")
                print("Return: {}".format(sum(rewards)))
                break

        return Episode(
                states=states,
                actions=actions,
                rewards=rewards
        )


"""
write monte carlo algo
incorporate noise
finish line intersection
make something to load tracks from csvs
modify this so that the car can drive in any direction, then it can do any track!!!
"""


if __name__ == "__main__":

    racetrack = RaceTrack(
            track_positions=[(k, k) for k in range(10)],
            start_positions=[(0, 0)],
            finish_positions=[(9, 9)],
    )

    car = Car(None, 5, 1)

    brain = Brain(car, racetrack)
    car.set_policy(brain.greedy_policy())

    run_epsiode(car, racetrack)

