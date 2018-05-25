#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 18:15
---------------------------------

"""
import os

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

import constants as c
from exercises import plotting

# Make the charts asked for in the thing
# also make some charts of how the values converge as the real ones move
# but for this you'll need the samples!


def load_file(name):
    return pd.read_pickle(
            os.path.join(c.Paths.exercise_output, 'ex_2_5', name),
    ).rename(columns=int)


if __name__ == '__main__':
    epsilon = 0.1
    estimator_type = 'ExponentialRecencyWeightedEstimator'.lower()

    all_exponential_choices = load_file(
            r'choices_{}_eps{}.pkl'.format(
                    'ExponentialRecencyWeightedEstimator'.lower(),
                    epsilon
            )
    )

    all_average_choices = load_file(
            r'choices_{}_eps{}.pkl'.format('sampleaverageestimator', epsilon)
    )

    all_optimal = load_file(r'optimal.pkl')

    perc_average_optimal = all_average_choices.eq(all_optimal).expanding().mean()
    perc_exponential_optimal = all_exponential_choices.eq(all_optimal).expanding().mean()

    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)
        ax.plot(perc_average_optimal.mean(1), label='Sample Average Method')
        ax.plot(perc_exponential_optimal.mean(1), label='Exponential Recency Weighted Method')
        print('ready')

        ax.grid(alpha=0.25)
        ax.legend(loc='lower right')
        ax.set_title('Comparison of Estimation Methods on 10-Bandit Test Bed')
        ax.set_xlabel(r'Number of Iterations')
        ax.set_ylabel(r'% Optimal Choices (Cumulative)')
        plt.tight_layout()
        fig.savefig(
                os.path.join(
                        c.Paths.exercise_output,
                        'ex_2_5',
                        'learning_curve.png'
                )
        )
