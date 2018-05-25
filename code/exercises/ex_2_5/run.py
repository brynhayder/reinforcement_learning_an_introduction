#!/usr/bin/env python
"""
--------------------------------
project: code
created: 10/04/2018 17:39
---------------------------------

"""
import os

import numpy as np
import pandas as pd

import constants as c
from bandits import SampleAverageEstimator, ExponentialRecencyWeightedEstimator, ActionValueBanditAgent, EpsilonGreedyActor
from bandits import utils

from concurrent.futures import ProcessPoolExecutor

# apparently these are thread safe so we are okay
AGENT_RANDOM_STATE = np.random.RandomState(seed=1)

N_STEPS = int(1e5)
N_BANDITS = 10
N_ITERS = 200
EPSILON = 0.1
ALPHA = 0.1
INITIAL_VALUE = 0.


def save_frame(frame, filename):
    return frame.to_pickle(
            os.path.join(
                    c.Paths.exercise_output,
                    'ex_2_5',
                    filename
            )
    )


def new_agent():
    return ActionValueBanditAgent(
            estimators=[
                SampleAverageEstimator(INITIAL_VALUE)
                # ExponentialRecencyWeightedEstimator(
                #         alpha=ALPHA,
                #         initial_value=INITIAL_VALUE
                # )
                for _ in range(N_BANDITS)
            ],
            actor=EpsilonGreedyActor(
                    epsilon=EPSILON,
                    n_actions=N_BANDITS,
                    random_state=AGENT_RANDOM_STATE
            )
    )


if __name__ == '__main__':
    EstimatorType = SampleAverageEstimator

    sampler = utils.RandomWalkingValueSampler(
            n_steps=N_STEPS,
            n_bandits=N_BANDITS,
            loc=0.,
            scale=0.01,
            random_state=np.random.RandomState(seed=0)
    )

    all_choices = list()
    all_explore = list()
    all_optimal = list()
    results = list()

    with ProcessPoolExecutor(4) as executor:
        for i in range(N_ITERS):
            print('Submitting', i)

            agent = new_agent()
            samples = sampler.sample(initial_values=np.zeros(N_BANDITS))
            results.append(executor.submit(utils.run_single, agent, samples))

    print('waiting on results')
    for future in results:
        output = future.result()
        all_choices.append(output.choices)
        all_explore.append(output.explore)
        all_optimal.append(output.optimal)

    all_choices = pd.DataFrame(np.c_[all_choices].T)
    all_explore = pd.DataFrame(np.c_[all_explore].T)
    all_optimal = pd.DataFrame(np.c_[all_optimal].T)

    save_frame(
            all_choices,
            r'choices_{}_eps{}.pkl'.format(
                    EstimatorType.__name__.lower(),
                    EPSILON
            )
    )

    save_frame(
            all_explore,
            r'explore_{}_eps{}.pkl'.format(
                    EstimatorType.__name__.lower(),
                    EPSILON
            )
    )

    save_frame(all_optimal, r'optimal.pkl')
