#!/usr/bin/env python
"""
--------------------------------
project: code
created: 06/06/2018 15:59
---------------------------------

This code is not threadsafe, due to the RandomStates being passed around
"""

# I decided to allow the car to have negative velocities so that it can go left if it wants.

# TODO: The printing sucks here. Change it to use some imshow of a matrix or something.

from collections import namedtuple
import copy
import itertools

from collections import OrderedDict

import numpy as np

Episode = namedtuple('Episode', ['states', 'actions', 'rewards'])


def in_box(point, c1, c2):
    """Is point in box with *opposite* corners c1 and c2"""
    c1x, c1y = c1
    c2x, c2y = c2
    x, y = point
    return min(c1x, c2x) <= x <= max(c1x, c2x) and min(c1y, c2y) <= y <= max(c1y, c2y)


def choose_from(seq, random_state):
    """to get around numpy interpreting list of tuples as an array"""
    return seq[random_state.choice(len(seq))]


class RaceTrack:
    """Two dimensional racetrack"""

    def __init__(self, track_positions, start_positions, finish_positions,
                 noise_level=0.1, transition_penalty=-1, crash_penalty=-1, random_state=None):
        self.track_positions = set(track_positions)
        self.start_positions = list(start_positions)
        self.finish_positions = set(finish_positions)
        self.noise_level = noise_level
        self.transition_penalty = transition_penalty
        self.crash_penalty = crash_penalty
        self.random_state = random_state or np.random.RandomState(seed=0)

        self.x_min, self.x_max, self.y_min, self.y_max = self._get_extremes()

        self.car_position = None
        self.previous_car_position = None
        self.car_velocity = None
        self.finished = None

        self.format_dict = {
            'car': 'c',
            'start': 's',
            'finish': 'f',
            'border': '+',
            'track': ' ',
            'sep': ' '
        }

    def _get_extremes(self):
        x, y = zip(*self.track_positions)
        return min(x), max(x), min(y), max(y)

    def new_race(self):
        self.car_position = self.random_starting_position()
        self.previous_car_position = self.car_position
        self.car_velocity = self._stationary()
        self.finished = self.just_finished(self.previous_car_position, self.car_position, self.car_velocity)
        return None

    def __enter__(self):
        self.new_race()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def _stationary():
        return 0, 0

    def _noise(self):
        if self.noise_level is None:
            return False
        return self.random_state.binomial(n=1, p=self.noise_level) == 1

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
        return None

    def just_finished(self, old_position, new_position, velocity):
        """Check if car has intersected a finish position on last move.
         Return the finish position if it has, else None."""
        if new_position in self.finish_positions:
            return new_position

        x, y = new_position
        vx, vy = velocity
        for pos in self.finish_positions:
            if in_box(pos, old_position, new_position):
                x_fin, y_fin = pos
                if vx == 0:
                    if y - y_fin == 0:
                        return pos
                elif y - y_fin == vy/vx * (x - x_fin):
                    return pos
        else:
            return None

    def on_track(self, old_position, new_position):
        x_new, y_new = new_position

        if not (self.x_min <= x_new <= self.x_max and self.y_min <= y_new <= self.y_max):
            return False

        x, y = old_position
        dx = x_new - x
        dy = y_new - y

        if dx == 0:
            points = [(x, i) for i in range(min(y, y_new), max(y, y_new)+1)]
        elif dy == 0:
            points = [(i, y) for i in range(min(x, x_new), max(x, x_new) + 1)]
        else:
            points = list()
            for i in range(min(x, x_new), max(x, x_new)):
                this_ymin = int(dy/dx * (i - x_new) + y_new)
                this_ymax = int(dy/dx * (i + 1 - x_new) + y_new)
                points.extend((i, y) for y in range(this_ymin, this_ymax))
            points.append(new_position)
        return all(p in self.track_positions for p in points)

    def random_starting_position(self):
        return choose_from(self.start_positions, self.random_state)

    def accelerate(self, dvx, dvy):
        dx, dy = self.car_velocity
        return self.set_velocity(dx + dvx, dy + dvy)

    # This thing has a side effect and a return value...
    def set_velocity(self, dx, dy):
        if self._noise():
            self.car_velocity = self._stationary()
        else:
            self.previous_car_position = self.car_position
            x, y = self.car_position
            new_position = x + dx, y + dy

            finishing_position = self.just_finished(
                    old_position=self.car_position,
                    new_position=new_position,
                    velocity=(dx, dy)
            )
            self.finished = finishing_position is not None

            if self.finished:
                self.car_position = finishing_position
                self.car_velocity = (dx, dy)
            elif self.on_track(self.previous_car_position, new_position):
                self.car_position = new_position
                self.car_velocity = (dx, dy)
            else:
                self.car_position = self.random_starting_position()
                self.car_velocity = self._stationary()
                return self.crash_penalty
        return self.transition_penalty  # reward for the transition

    def __str__(self):
        return self.track_string(self.format_dict)

    def track_string(self, format_dict):
        x, y = zip(*self.track_positions)
        output = ""
        y_rng = range(max(y) + 1)
        for i in range(max(x) + 1):
            row = ""
            for j in y_rng:
                pos = i, j
                if pos == self.car_position:
                    row += format_dict['car']
                elif pos in self.start_positions:
                    row += format_dict['start']
                elif pos in self.finish_positions:
                    row += format_dict['finish']
                elif pos in self.track_positions:
                    row += format_dict['track']
                else:
                    row += format_dict['border']
                row += format_dict['sep']
            output = row + "\n" + output
        return output

    def episode_string(self, episode):
        """
        Print the racetrack with car positions labelled by latest timestep
        that the car was in that position.

        Args:
            episode (Episode):

        Returns:
            (str):
        """
        cp, _ = zip(*episode.states)

        car_positions = dict()
        for i, p in enumerate(cp):
            car_positions[p] = i

        x, y = zip(*self.track_positions)
        output = ""
        y_rng = range(max(y) + 1)
        for i in range(max(x) + 1):
            row = ""
            for j in y_rng:
                pos = i, j
                if pos in car_positions:
                    row += str(car_positions[pos])
                elif pos in self.start_positions:
                    row += self.format_dict['start']
                elif pos in self.finish_positions:
                    row += self.format_dict['finish']
                elif pos in self.track_positions:
                    row += self.format_dict['track']
                else:
                    row += self.format_dict['border']
                row += self.format_dict['sep']
            output = row + "\n" + output
        return output

    def print_episode(self, episode):
        return print(
                self.episode_string(episode=episode)
        )


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
            if -self.speed_limit <= dx + ax <= self.speed_limit
               and -self.speed_limit <= dy + ay <= self.speed_limit
               and not (v == (0, 0) == (ax, ay))
        ]

    # When we generate the possible states here we are not completely accurate, in that
    # we generate some unreachable states (i.e. having max velocity after 1 timestep).
    # None of these states are reachable in an episode, so their values won't be updated
    # and the policy in those places is irrelevant.
    def possible_states(self, track):
        return list(
                itertools.product(
                    track.track_positions,
                    itertools.product(range(-self.speed_limit, self.speed_limit + 1), repeat=2)
                )
        )

    def set_policy(self, policy):
        self.policy = policy
        return None

    def drive(self, current_position, current_velocity):
        return self.policy[current_position, current_velocity]


class Brain:
    @staticmethod
    def initial_action_values(car, racetrack):
        # One should not initialise the action values to a value that is
        # larger than the largest possible return on the track (since then it is possible that the
        # algorithm will cease to learn anything if there is an action that is necessary to termination).
        # We therefore initialise the values to the worst possible return for the track (without going off).
        # Note that if the reward dynamics are changed for this problem, then this will need to be checked.
        init = - len(racetrack.track_positions) ** 2
        return {
            s: OrderedDict((a, init) for a in car.possible_actions(s))
            for s in car.possible_states(racetrack) if s[0] not in racetrack.finish_positions
        }

    def __init__(self, car, racetrack, epsilon=0.1, random_state=None):
        self.car = car
        self.racetrack = racetrack
        self.epsilon = epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

        self.action_values = self.initial_action_values(car, racetrack)
        self._normalising_factor = {
            s: {
                a: 0 for a in v
            }
            for s, v in self.action_values.items()
        }

    def greedy_policy(self):
        return {
            k: greedy(a) for k, a in self.action_values.items()
        }

    def epsilon_greedy_policy(self):
        return EpsilonGreedy(
                action_values=copy.deepcopy(self.action_values),
                epsilon=self.epsilon,
                random_state=self.random_state
        )

    # note that the update only work for epsilon greedy policies because of the last line.
    # probably need a policy class to deal with this.
    def update(self, episode):
        """

        Args:
            episode (Episode): The data for the episode
        """
        returns = 0
        w = 1
        for s, a, r in zip(reversed(episode.states[:-1]), reversed(episode.actions), reversed(episode.rewards)):
            returns += r
            self._normalising_factor[s][a] += w
            self.action_values[s][a] += w / self._normalising_factor[s][a] * (returns - self.action_values[s][a])

            if a != greedy(self.action_values[s]):
                break

            w *= 1 - self.epsilon + self.epsilon / len(self.action_values[s])
        return None


def greedy(state_action_values):
    return max(state_action_values, key=lambda k: state_action_values[k])


class EpsilonGreedy:
    def __init__(self, action_values, epsilon, random_state):
        self.action_values = action_values
        self.epsilon = epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

    def explore(self):
        return self.random_state.binomial(n=1, p=self.epsilon) == 1

    def __getitem__(self, state):
        if self.explore():
            return choose_from(list(self.action_values[state].keys()), self.random_state)
        else:
            return greedy(self.action_values[state])


def run_epsiode(car, racetrack, start_position=None, print_=False):
    """

    Args:
        car (Car):
        racetrack (RaceTrack):

    Returns:
        Episode (Episode): The trajectory.
    """
    with racetrack as race:
        if start_position is not None:
            race.car_position = start_position
        states = [(race.car_position, race.car_velocity)]
        actions = list()
        rewards = list()

        while not race.finished:
            action = car.drive(
                    current_position=race.car_position,
                    current_velocity=race.car_velocity
            )

            if print_:
                print(f"State: {race.car_position}\nAction: {action}")

            reward = race.accelerate(*action)
            actions.append(action)
            rewards.append(reward)
            states.append((race.car_position, race.car_velocity))

        return Episode(
                states=states,
                actions=actions,
                rewards=rewards
        )


def train(brain, car, racetrack, runs_per_pos=1):
    r = 0
    for pos in racetrack.start_positions:
        for _ in range(runs_per_pos):
            e = run_epsiode(car, racetrack, start_position=pos)
            brain.update(e)
            r += sum(e.rewards)
    return r / len(racetrack.start_positions) / runs_per_pos


if __name__ == "__main__":
    EPS = 0.1
    N_RUNS = 10 ** 4
    size = 10

    track_positions = [(k, k) for k in range(size)]

    start_positions = [
        (0, 0),
    ]

    finish_positions = [
        (size - 1, size - 1),
    ]

    for pos in start_positions:
        if pos not in track_positions:
            track_positions.append(pos)
    for pos in finish_positions:
        if pos not in track_positions:
            track_positions.append(pos)

    racetrack = RaceTrack(
            track_positions=track_positions,
            start_positions=start_positions,
            finish_positions=finish_positions,
            noise_level=0.1
    )

    car = Car(None, 5, 1)
    brain = Brain(car, racetrack, epsilon=EPS)

    print(racetrack)

    episodes = list()

    for i in range(N_RUNS):
        behaviour_policy = brain.epsilon_greedy_policy()
        car.set_policy(behaviour_policy)
        e = run_epsiode(car, racetrack)
        brain.update(e)
        episodes.append(e)
        if i % 1000 == 0:
            print(f"Finished episode {i}")
            print(f"Return: {sum(e.rewards)}")
            racetrack.print_episode(e)
            print(e.states)

    print("\n\n")
    racetrack.set_noise_level(None)
    car.set_policy(brain.greedy_policy())
    greedy_episode = run_epsiode(car, racetrack)
    print("Greedy Episode")
    print(f"Return: {sum(greedy_episode.rewards)}")
    racetrack.print_episode(greedy_episode)
    print(greedy_episode.states)
