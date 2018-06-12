#!/usr/bin/env python
"""
--------------------------------
project: code
created: 12/06/2018 13:38
---------------------------------

"""

import numpy as np


def load_track(path, track_flag=0, start_flag=2, finish_flag=3, delimiter=',', dtype=int):
    arr = np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=dtype
    )
    return parse_track(
            arr,
            track_flag=track_flag,
            start_flag=start_flag,
            finish_flag=finish_flag
    )


def parse_track(arr, track_flag, start_flag, finish_flag):
    track_positions = get_locs(arr, track_flag)
    start_positions = sorted(get_locs(arr, start_flag))
    finish_positions = sorted(get_locs(arr, finish_flag))

    track_positions.extend(start_positions)
    track_positions.extend(finish_positions)
    return dict(
            track_positions=sorted(track_positions),
            start_positions=start_positions,
            finish_positions=finish_positions,
    )


def get_locs(arr, value):
    x, y = np.where(arr == value)
    x = arr.shape[0] - 1 - x
    return list(zip(x, y))