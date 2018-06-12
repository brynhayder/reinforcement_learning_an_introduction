#!/usr/bin/env python
"""
--------------------------------
project: code
created: 12/06/2018 13:41
---------------------------------

"""
import pickle
import os

from matplotlib import colors, use; use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import constants as c
import plotting

from exercises.ex_5_10.model import RaceTrack
from exercises.ex_5_10.utils import load_track


def plot_learning_curve(returns, epsilons, track_name):
    fig, ax = plt.subplots(1)
    ax.plot(-np.array(returns), label="$-G$")
    ax.set_yscale('log')
    ax.set_ylabel(r"$-1 \times$ Returns (log scale)")
    ax.set_xlabel("Episode")

    ax1 = ax.twinx()
    ax1.plot(epsilons, label="$\epsilon$ (right)", color='C1')

    ax.set_title(f"Learning Curve: {track_name}")

    plotting.multi_ax_legend(ax, ax1)
    ax1.grid(alpha=0.1, which="both", ls=':')
    return fig, (ax, ax1)


def grid(racetrack, episode, format_dict):
    def replace(row, dct):
        return [dct[r] if r in dct else int(r) for r in row]

    s = racetrack.episode_string(episode)

    rows = [
        replace([i for i in k.split(racetrack.format_dict['sep']) if i], format_dict)
        for k in s.split('\n')
    ]
    return np.array(list(filter(bool, rows)))


def plot_trajectory(ax, racetrack, episode):
    racetrack.format_dict['track'] = "aa"  # because track and sep are the same

    format_dict = {
        racetrack.format_dict['track']: -1,
        racetrack.format_dict['start']: -2,
        racetrack.format_dict['finish']: -3,
        racetrack.format_dict['border']: -4
    }

    e_grid = grid(episode=episode, racetrack=racetrack, format_dict=format_dict)

    cs = ['gray', 'C2', 'C3', 'white']
    bounds = list(range(-4, 0))
    for i in range(len(episode.states)):
        cs.append('C0')
        bounds.append(i)

    cmap = colors.ListedColormap(cs)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(e_grid, cmap=cmap, norm=norm)

    def make_ticks(rng):
        return [k - 0.5 for k in rng]

    ax.set_xticks(make_ticks(list(range(e_grid.shape[1]))[::-1]))
    ax.set_yticks(make_ticks(range(e_grid.shape[0])))

    ax.grid(alpha=0.05, which='both', lw=0.1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0, width=0)
    return None


if __name__ == "__main__":
    TRACK_NAME = "track_2"

    with open(os.path.join(c.Paths.output, 'ex_5_10', f'{TRACK_NAME}.pkl'), 'rb') as f:
        info = pickle.load(f)

    fig, axarr = plot_learning_curve(
            info['returns'],
            info['training_epsilons'],
            track_name=TRACK_NAME.replace('_', ' ').title()
    )

    fig.savefig(
            os.path.join(
                    c.Paths.output,
                    "ex_5_10",
                    f"{TRACK_NAME}_learning_curve.eps"
            ),
            format='eps',
            dpi=1000
    )

    track_points = load_track(
            os.path.join(
                    c.Paths.input,
                    'ex_5_10',
                    f"{TRACK_NAME}.csv"
            )
    )

    racetrack = RaceTrack(
            **track_points
    )

    # with plt.rc_context(plotting.rc()):
    #     fig, axarr = plt.subplots(2, 3)
    #     for (p, e), ax in zip(info['greedy_episodes'].items(), axarr.flatten()):
    #         plot_trajectory(ax, racetrack, e)
    #         ax.set_title(f"Start: {p}. Return {sum(e.rewards)}", fontsize=10)
    #     fig.suptitle(f"{TRACK_NAME.replace('_', ' ').title()} Trajectories", fontsize=16)
    #     fig.savefig(
    #             os.path.join(
    #                     c.Paths.output,
    #                     "ex_5_10",
    #                     f"{TRACK_NAME}_trajectories.eps"
    #             ),
    #             format='eps',
    #             dpi=1000
    #     )

    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)
        p = (0, 3)
        e = info['greedy_episodes'][p]
        plot_trajectory(ax, racetrack, e)
        ax.set_title(f"Start: {p}. Return {sum(e.rewards)}", fontsize=10)
        fig.suptitle(f"{TRACK_NAME.replace('_', ' ').title()} Sample Trajectory", fontsize=16)
        fig.savefig(
                os.path.join(
                        c.Paths.output,
                        "ex_5_10",
                        f"{TRACK_NAME}_sample_trajectory.eps"
                ),
                format='eps',
                dpi=1000
        )

        # plt.show()
