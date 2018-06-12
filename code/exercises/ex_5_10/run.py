#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/06/2018 18:00
---------------------------------

I added a crash penalty of -100 to deter the agent from
running into walls to end up at favourable starting positions.

Also, the algorithm doesn't seem to converge well with the noise on.

"""
import pickle
import os

import numpy as np

import constants as c

from exercises.ex_5_10.model import Brain, Car, RaceTrack, run_epsiode, train
from exercises.ex_5_10.utils import load_track


if __name__ == "__main__":

    TRACK_NAME = "track_2"
    eps_start = 0.1

    folder = os.path.join(
            c.Paths.input,
            'ex_5_10'
    )

    track_indices = load_track(
            path=os.path.join(folder, f"{TRACK_NAME}.csv"),
            track_flag=0,
            start_flag=2,
            finish_flag=3
    )

    racetrack = RaceTrack(
            noise_level=None, #0.1,
            crash_penalty=-1000,
            **track_indices
    )
    car = Car(None, 5, 1)

    brain = Brain(
            car,
            racetrack,
            epsilon=eps_start,
            random_state=np.random.RandomState(seed=123)
    )

    print(racetrack)

    # initialise the policy with random runs
    brain.epsilon = 1.
    for i in range(3):
        car.set_policy(
                brain.epsilon_greedy_policy()
        )
        g = train(brain, car, racetrack)
        print("------------------------------------------------------")
        print(f"Finished random policy episode set {i}")
        print(f"Epsilon = {brain.epsilon}")
        print(f"Average Return: {g}")
        print("------------------------------------------------------")
        print("\n")

    brain.epsilon = eps_start
    returns = list()
    training_epsilons = list()
    n_runs = 20
    for i in range(n_runs):
        car.set_policy(
                brain.epsilon_greedy_policy()
        )
        g = train(brain, car, racetrack)
        returns.append(g)
        training_epsilons.append(brain.epsilon)
        print("------------------------------------------------------")
        print(f"Finished episode set {i}")
        print(f"Epsilon = {brain.epsilon}")
        print(f"Average Return: {g}")
        print("------------------------------------------------------")
        print("\n")
        # brain.epsilon -= eps_start / n_runs

    greedy_episodes = dict()
    print("\n")
    racetrack.set_noise_level(None)
    car.set_policy(brain.greedy_policy())
    for pos in racetrack.start_positions:
        greedy_episode = run_epsiode(
                car,
                racetrack,
                start_position=pos
        )
        print(f"Greedy Episode: starting at {pos}")
        print(f"Return: {sum(greedy_episode.rewards)}")
        racetrack.print_episode(greedy_episode)
        greedy_episodes[pos] = greedy_episode

    info = dict(
            track_name=TRACK_NAME,
            returns=returns,
            training_epsilons=training_epsilons,
            greedy_episodes=greedy_episodes
    )

    with open(os.path.join(c.Paths.output, 'ex_5_10', f'{TRACK_NAME}.pkl'), 'wb') as f:
        pickle.dump(info, f)

