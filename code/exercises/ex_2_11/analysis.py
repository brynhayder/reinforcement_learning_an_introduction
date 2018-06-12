#!/usr/bin/env python
"""
--------------------------------
project: code
created: 13/04/2018 14:43
---------------------------------

"""
import os

import matplotlib; matplotlib.use('svg')
import matplotlib.pyplot as plt
import pandas as pd

import constants as c
import plotting

if __name__ == '__main__':
    samples = pd.read_pickle(
            os.path.join(
                    c.Paths.output,
                    'ex_2_11',
                    'samples.pkl'
            )
    )

    results = pd.read_pickle(
            os.path.join(
                    c.Paths.output,
                    'ex_2_11',
                    'results.pkl'
            )
    )

    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)
        samples.plot(ax=ax)
        ax.legend(
                title='Actions',
                bbox_to_anchor=(1, 1),
                loc='upper left'
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel('$t$')
        ax.set_ylabel('Action Values')
        ax.set_title('True Action Values on 10-Armed Bandit')
        # plt.tight_layout()
        fig.savefig(
                os.path.join(
                    c.Paths.output,
                    'ex_2_11',
                    'action_values.png'
                ),
                bbox_inches='tight'
        )

        fig, ax = plt.subplots(1)
        results.plot(ax=ax)
        ax.set_xscale('log', basex=2)
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Proportion Optimal Choice')
        ax.set_title(r'Parameter Study of $\varepsilon$-greedy Action Value Agent on 10-Armed Test Bed')
        ax.grid(alpha=0.25)
        # plt.tight_layout()
        fig.savefig(
                os.path.join(
                    c.Paths.output,
                    'ex_2_11',
                    'parameter_study.png'
                ),
                bbox_inches='tight'
        )

