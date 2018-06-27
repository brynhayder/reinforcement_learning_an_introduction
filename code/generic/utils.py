#!/usr/bin/env python
"""
--------------------------------
project: code
created: 26/06/2018 17:29
---------------------------------

"""


def choose_from(seq, random_state):
    """to get around numpy interpreting list of tuples as an array"""
    return seq[random_state.choice(len(seq))]