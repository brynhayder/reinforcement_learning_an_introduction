#!/usr/bin/env python
"""
--------------------------------
project: code
created: 24/07/2018 11:47
---------------------------------

"""
import os

import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import plotting
import constants as c


def q(state, w):
    if state < 6:
        return 2 * w[state] + w[-1]
    else:
        return w[state] + 2 * w[-1]


def feature(state):
    out = np.zeros(8)
    if state < 6:
        out[state] = 2
        out[-1] = 1
        return out
    else:
        out[state] = 1
        out[-1] = 2
        return out


if __name__ == "__main__":
    n_steps = 250
    random_state = np.random.RandomState(seed=0)

    states = np.arange(7)
    weights = np.array([1, 1, 1, 1, 1, 1, 10, 1])
    alpha = 0.01
    gamma = 0.99

    weights_list = [weights]
    for i in range(n_steps):
        s = random_state.choice(states)
        weights = weights + 7 * alpha * (gamma * q(states[-1], weights) - q(s, weights)) * feature(s)
        weights_list.append(weights)

    output = np.c_[weights_list]

    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)
        lines = ax.plot(output)
        ax.legend(lines, [f"w{i+1}" for i in range(output.shape[1])])
        ax.grid(alpha=0.1)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Weight")
        ax.set_title("Q-learning on Baird's Counterexample")
        plt.tight_layout()

        plotting.savefig(
                fig,
                path=os.path.join(
                        c.Paths.output,
                        "ex_11_3",
                        "bairds_counter_example_q_learning.png"
                )
        )






