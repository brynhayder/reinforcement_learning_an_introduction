#!/usr/bin/env python
"""
--------------------------------
project: code
created: 12/04/2018 12:47
---------------------------------

"""


def rc(kwds={}):
    default_kwds = {
        'figure.figsize': (20, 10),
        'font.size': 12
    }
    default_kwds.update(kwds)
    return kwds
