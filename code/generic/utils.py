#!/usr/bin/env python
"""
--------------------------------
project: code
created: 26/06/2018 17:29
---------------------------------

"""
from collections import namedtuple

Results = namedtuple('Results', ['states', 'actions', 'rewards'])


def choose_from(seq, random_state):
    """to get around numpy interpreting list of tuples as an array"""
    return seq[random_state.choice(len(seq))]


def run_episode(agent, environment, maxiter=10 ** 5, update=True):
    environment.reset()

    states = [environment.current_state]
    actions = list()
    rewards = list()
    for _ in range(maxiter):
        old_state = environment.current_state
        action = agent.choose_action(state=old_state)
        new_state, reward, done = environment.step(action)

        if update:
            agent.update(old_state, action, reward, new_state)

        states.append(new_state)
        actions.append(action)
        rewards.append(reward)

        if done:
            break

    return Results(
            states=states,
            rewards=rewards,
            actions=actions
    )


def run_continuous(agent, environment, n_steps, update=True):
    environment.reset()

    states = [environment.current_state]
    actions = list()
    rewards = list()
    for _ in range(n_steps):
        old_state = environment.current_state
        action = agent.choose_action(state=old_state)
        new_state, reward, done = environment.step(action)

        if update:
            agent.update(old_state, action, reward, new_state)

        actions.append(action)
        rewards.append(reward)
        states.append(new_state)

        if done:
            environment.reset()

    return Results(
            states=states,
            actions=actions,
            rewards=rewards
    )
