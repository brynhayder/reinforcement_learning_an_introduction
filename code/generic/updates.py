#!/usr/bin/env python
"""
--------------------------------
project: code
created: 27/06/2018 11:43
---------------------------------

update action-value dictionaries in-place

"""


def sarsa(action_values, old_state, action, reward, new_state, new_action, alpha, gamma):
    """update action values in-place"""
    action_values[old_state][action] += alpha * (
            reward + gamma * action_values[new_state][new_action] - action_values[old_state][action]
    )
    return None
