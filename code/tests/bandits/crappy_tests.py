#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 15:15
---------------------------------

These tests are very informal

"""
import unittest

import numpy as np

from bandits import EpsilonGreedyActor, SampleAverageEstimator, ActionValueBanditAgent


class EpsilonGreedyActorTestCase(unittest.TestCase):
    def test_proportion_exploration(self):
        epsilon = 0.01

        actor = EpsilonGreedyActor(
                n_actions=4,
                random_state=np.random.RandomState(None),
                epsilon=epsilon
        )

        optimal_action = 3
        n = 10000

        choices = list()
        for i in range(n):
            choices.append(actor.action([optimal_action]))

        prop = np.sum(np.array(choices) == optimal_action)

        print(1 - prop/n)
        print(epsilon)
        return None


class SampleAverageEstimatorTestCase(unittest.TestCase):
    def test_convergence(self):
        mean = 0.98

        state = np.random.RandomState(0)
        samples = state.normal(loc=mean, scale=1, size=int(1e5))

        estimator = SampleAverageEstimator(default_value=0)

        for x in samples:
            estimator.update(x)

        print('Estimator value is', estimator.value)
        print('Sample mean is', np.mean(samples))
        print('True mean is', mean)
        return None


class BanditAgentTestCase(unittest.TestCase):
    pass
    # CHANGE THIS INTO SOME SHORT TEST

    # print(np.sum(choices == optimal_actions) / n_steps)
    #
    # print(samples)
    #
    # print(np.c_[[choices, optimal_actions, explore]].T)

    #
    # print('------------------------------')
    # print('Results')
    # print('------------------------------')
    # print('{:<10} {:<10} {:<10}'.format('choice', 'reward', 'optimal'))
    # print('------------------------------')
    # choices = list()
    # for row in samples:
    #     choice = agent.action()
    #     choices.append(choice)
    #     reward = row[choice]
    #     agent.update(choice, reward)
    #     optimal = np.argmax(row)
    #     print('{:<10} {:<10} {:<10}'.format(choice, '{:.2f}'.format(reward), optimal))
    # print('------------------------------')



    # write a small test for the agent thing
    # figure out a way to do the analysis


    # agent.update(0, 10)
    # print(agent.get_estimates())
    # print(agent.action())
    #
    # agent.update(0, 1)
    # print(agent.get_estimates())
    # print(agent.action())
    #
    # e = agent.estimators[0]
    # print(e.n_updates)
    # n_bandits = 1 #10
    # n_steps = int(1e5)
    # sampler = RandomWalkingValueSampler(n_bandits=n_bandits, n_steps=n_steps)
    # samples = sampler(initial_values=np.zeros(n_bandits))

if __name__ == '__main__':
    unittest.main()
