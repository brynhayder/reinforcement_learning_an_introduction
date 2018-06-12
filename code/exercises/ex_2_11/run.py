#!/usr/bin/env python
"""
--------------------------------
project: code
created: 13/04/2018 14:01
---------------------------------

"""
from concurrent.futures import ProcessPoolExecutor
import os

import matplotlib; matplotlib.use('TKAgg')
import numpy as np
import pandas as pd

from bandits import (
    ExponentialRecencyWeightedEstimator,
    SampleAverageEstimator,
    EpsilonGreedyActor,
    ActionValueBanditAgent,
    utils
)
import constants as c


N_STEPS = 200000
N_BANDITS = 10
PARAMS = 2. ** np.arange(-7, -1)
INITIAL_VALUE = 0.
STEP_SIZE = 0.1


def process_outputs(output):
    grades = output.choices == output.optimal
    _, second_half = np.array_split(grades, 2)
    return np.mean(second_half)


def evaluate_single_agent(agent, samples):
    return process_outputs(
            utils.run_single(
                    agent,
                    samples
            )
    )


if __name__ == "__main__":
    sampler = utils.RandomWalkingValueSampler(
            n_steps=N_STEPS,
            n_bandits=N_BANDITS,
            loc=0.,
            scale=0.01,
            random_state=np.random.RandomState(seed=0)
    )

    samples = sampler.sample(initial_values=np.zeros(N_BANDITS))

    sample_average_outputs = dict()
    constant_step_outputs = dict()
    with ProcessPoolExecutor(4) as executor:
        for param in PARAMS:
            print('Sumbitting', param)

            sample_average_agent = ActionValueBanditAgent(
                    estimators=[
                        SampleAverageEstimator(INITIAL_VALUE)
                        for _ in range(N_BANDITS)
                    ],
                    actor=EpsilonGreedyActor(
                            epsilon=param,
                            n_actions=N_BANDITS,
                            random_state=np.random.RandomState(seed=1)
                    )
            )

            sample_average_outputs[param] = executor.submit(evaluate_single_agent, sample_average_agent, samples)

            constant_step_agent = ActionValueBanditAgent(
                    estimators=[
                        ExponentialRecencyWeightedEstimator(step_size=STEP_SIZE, initial_value=INITIAL_VALUE)
                        for _ in range(N_BANDITS)
                    ],
                    actor=EpsilonGreedyActor(
                            epsilon=param,
                            n_actions=N_BANDITS,
                            random_state=np.random.RandomState(seed=1)
                    )
            )

            constant_step_outputs[param] = executor.submit(evaluate_single_agent, constant_step_agent, samples)

    print('Waiting on results')
    sample_average_outputs = {k: v.result() for k, v in sample_average_outputs.items()}
    constant_step_outputs = {k: v.result() for k, v in constant_step_outputs.items()}

    results = pd.concat(
                    [
                        pd.Series(sample_average_outputs, name='Sample Average'),
                        pd.Series(constant_step_outputs, name='Constant Step')
                    ],
                    axis=1
            )

    pd.DataFrame(samples).to_pickle(
            os.path.join(
                    c.Paths.output,
                    'ex_2_11',
                    'samples.pkl'
            )
    )

    results.to_pickle(
            os.path.join(
                    c.Paths.output,
                    'ex_2_11',
                    'results.pkl'
            )
    )
