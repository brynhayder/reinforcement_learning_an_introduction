#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 18:02
---------------------------------

"""
import os


class Paths(object):
    data = os.path.abspath(
            os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir,
                    'data'
            )
    )

    output = os.path.join(data, 'exercise_output')
    input = os.path.join(data, 'exercise_input')
