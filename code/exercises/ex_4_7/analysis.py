#!/usr/bin/env python
"""
--------------------------------
project: code
created: 24/05/2018 20:50
---------------------------------

"""
from itertools import product
import os

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from plotting import rc
from exercises.utils import read_pickle
from exercises.ex_4_7 import output_folder

if __name__ == "__main__":
    plt.rcParams.update(rc())
    plt.rcParams.update({'figure.figsize': (15, 8)})
    policy = read_pickle(os.path.join(output_folder, 'policy.pkl'))
    values = read_pickle(os.path.join(output_folder, 'values.pkl'))

    max_cars = values.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    lim = np.max(np.abs(policy))
    ax.matshow(policy.T, cmap=plt.cm.bwr, vmin=-lim, vmax=lim)
    ax.set_xticks(range(max_cars))
    ax.set_yticks(range(max_cars))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel("Cars at location x")
    ax.set_ylabel("Cars at location y")
    ax.set_xticks([x - 0.5 for x in range(1, max_cars)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, max_cars)], minor=True)
    for x, y in product(range(max_cars), range(max_cars)):
        ax.text(x=x, y=y, s=int(policy[x, y]), va='center', ha='center', fontsize=8)
    ax.set_title(r'$\pi_*$', fontsize=20)

    x, y = zip(*product(range(max_cars), repeat=2))
    surface = [values[i, j] for i, j in zip(x, y)]
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter3D(x, y, surface)
    ax.set_xlim3d(0, max_cars)
    ax.set_ylim3d(0, max_cars)
    ax.set_xlabel("Cars at location x")
    ax.set_ylabel("Cars at location y")
    ax.set_title('$v_*$', fontsize=20)

    plt.savefig(
            os.path.join(output_folder, 'altered_car_rental.png'),
            bbox_inches='tight'
    )
    # plt.show()
