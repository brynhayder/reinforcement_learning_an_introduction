#!/usr/bin/env python
"""
--------------------------------
project: code
created: 25/05/2018 11:56
---------------------------------

"""
import copy
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from functools import wraps

import numpy as np
from scipy.stats import poisson as _poisson

N_PROCS = 3


def cache_on_instance(func):
    """caching for single argument methods"""
    @wraps(func)
    def wrapper(instance, n):
        cache = getattr(instance, "{}_cache".format(func.__name__))
        if n not in cache:
            output = func(instance, n)
            cache[n] = output
            return output
        else:
            return func(instance, n)
    return wrapper


class Poisson:
    """Caching Poisson Distribution"""
    def __init__(self, mean):
        self.mean = mean

        self.pmf_cache = dict()
        self.cdf_cache = dict()

    @cache_on_instance
    def pmf(self, n):
        return np.exp(- self.mean) * np.power(self.mean, n) / np.math.factorial(n)
        # return _poisson.pmf(n, self.mean)

    @cache_on_instance
    def cdf(self, n):
        return _poisson.cdf(n, self.mean)


class Distributions:
    def __init__(self):
        self.x_rental = Poisson(3)
        self.x_return = Poisson(3)
        self.y_rental = Poisson(4)
        self.y_return = Poisson(2)


class Environment:
    def __init__(self, distributions, max_cars, max_car_moves, rental_reward, movement_cost, populate_dynamics=True):
        self.distributions = distributions
        self.max_cars = max_cars
        self.max_car_moves = max_car_moves
        self.rental_reward = rental_reward
        self.movement_cost = movement_cost

        self.expected_rewards = dict()
        self.transition_probabilities = dict()
        if populate_dynamics:
            self.populate_dynamics()

    def possible_states(self):
        return product(range(self.max_cars + 1), repeat=2)

    def possible_actions(self, state):
        x, y = state
        return range(
                -min(y, self.max_car_moves, self.max_cars - x),
                min(x, self.max_car_moves, self.max_cars - y) + 1
        )

    @staticmethod
    def expected_rentals(initial_cars, distribution):
        return sum(
                distribution.pmf(i) * i for i in range(initial_cars + 1)
        ) + (1 - distribution.cdf(initial_cars)) * initial_cars

    def expected_reward(self, state, action):
        x, y = state
        rental_reward = self.rental_reward * (
                self.expected_rentals(x - action, self.distributions.x_rental)
                + self.expected_rentals(y + action, self.distributions.y_rental)
        )
        return rental_reward - self.movement_cost * abs(action)

    def _single_location_transition_probability(self, final_cars, initial_cars, rental_dist, return_dist):
        prob = 0
        min_rentals = max(initial_cars - final_cars, 0)
        max_rentals = initial_cars
        for i in range(min_rentals, max_rentals + 1):
            term = return_dist.pmf(final_cars - initial_cars + i)
            if final_cars == self.max_cars:
                term += (1 - return_dist.cdf(final_cars - initial_cars + i))
            prob += rental_dist.pmf(i) * term
        prob += (1 - rental_dist.cdf(initial_cars)) * return_dist.pmf(final_cars)
        return prob

    def transition_probability(self, new_state, old_state, action):
        x, y = old_state
        x_new, y_new = new_state
        return self._single_location_transition_probability(
                final_cars=x_new,
                initial_cars=x - action,
                rental_dist=self.distributions.x_rental,
                return_dist=self.distributions.x_return
        ) * self._single_location_transition_probability(
                final_cars=y_new,
                initial_cars=y + action,
                rental_dist=self.distributions.y_rental,
                return_dist=self.distributions.y_return
        )

    @staticmethod
    def _calculate_single_state(tup):
        instance, state_ = tup
        inner_probs = dict()
        for action in instance.possible_actions(state_):
            for new_state in instance.possible_states():
                inner_probs[(new_state, state_, action)] = instance.transition_probability(new_state, state_, action)
        return inner_probs

    def _calculate_transition_probabilities(self):
        with ProcessPoolExecutor(N_PROCS) as executor:
            outputs = executor.map(self._calculate_single_state, [(self, s) for s in self.possible_states()])

        out = dict()
        for thing in outputs:
            out.update(thing)
        return out

    def _calculate_expected_rewards(self):
        rewards = dict()
        for s in self.possible_states():
            for a in self.possible_actions(s):
                rewards[(s, a)] = self.expected_reward(s, a)
        return rewards

    def populate_dynamics(self):
        self.transition_probabilities = self._calculate_transition_probabilities()
        self.expected_rewards = self._calculate_expected_rewards()
        return None


def expected_return(state, action, values, discount_factor, environment):
    output = sum(
            environment.transition_probabilities[(ns, state, action)] * values[ns]
            for ns in environment.possible_states()
    )
    return environment.expected_rewards[(state, action)] + discount_factor * output


def evaluate_policy(policy, values, tolerance, discount_factor, environment):
    values = copy.deepcopy(values)
    delta = tolerance
    sweep = 0
    while delta >= tolerance:
        delta = 0
        for state in environment.possible_states():
            v = values[state]
            values[state] = expected_return(
                    state=state,
                    action=policy[state],
                    values=values,
                    discount_factor=discount_factor,
                    environment=environment
            )
            delta = max(delta, abs(v - values[state]))
        sweep += 1
        print(f"Policy Evaluation: end of sweep {sweep}, delta = {delta}")
    return values


def greedy_action(state, values, discount_factor, environment):
    actions = environment.possible_actions(state)
    expected_returns = [expected_return(state, action, values, discount_factor, environment) for action in actions]
    return actions[np.argmax(expected_returns)]


def greedy_policy(values, discount_factor, environment):
    policy, _ = initial_policy_values(environment)
    for state in environment.possible_states():
        policy[state] = greedy_action(state, values, discount_factor, environment)
    return policy


def is_policy_stable(old_policy, new_policy):
    return np.all(old_policy == new_policy)


def policy_iteration(policy, values, environment, discount_factor, tolerance=0.01, maxiter=10**4):
    policy_stable = False
    iteration = 1
    while iteration <= maxiter and not policy_stable:
        print(f"POLICY EVALUATION: Iteration {iteration}")
        values = evaluate_policy(
                values=values,
                policy=policy,
                tolerance=tolerance,
                discount_factor=discount_factor,
                environment=environment
        )
        print('Improving Policy...')
        new_policy = greedy_policy(
                values=values,
                discount_factor=discount_factor,
                environment=environment
        )
        policy_stable = is_policy_stable(policy, new_policy)
        policy = new_policy
        print("========== POLICY ITERATION ==========")
        print(f"Iteration {iteration}: policy_stable = {policy_stable}")
        print("======================================")
        iteration += 1
    return policy, values


def initial_policy_values(environment):
    shape = environment.max_cars + 1, environment.max_cars + 1
    return np.zeros(shape, dtype=int), np.zeros(shape, dtype=float)


if __name__ == "__main__":
    import os
    import pickle
    from pprint import pprint
    import time

    from exercises.ex_4_7.jacks_car_rental import folder as output_folder

    MAX_CARS = 20
    MAX_CAR_MOVES = 5
    DISCOUNT_FACTOR = 0.9
    THRESHOLD = 1e-2
    MAXITER = 10**4

    RENTAL_REWARD = 10
    MOVEMENT_COST = 2

    print("Building environment")
    start = time.time()
    environment = Environment(
            distributions=Distributions(),
            max_cars=MAX_CARS,
            max_car_moves=MAX_CAR_MOVES,
            rental_reward=RENTAL_REWARD,
            movement_cost=MOVEMENT_COST,
            populate_dynamics=True
    )

    print("Building environment took {:.2f}".format(time.time() - start))

    with open(os.path.join(output_folder, f'environment_max_cars_{MAX_CARS}_max_moves_{MAX_CAR_MOVES}.pkl'), 'wb') as f:
        pickle.dump(environment, f)

    initial_policy, initial_values = initial_policy_values(environment)

    policy, values = policy_iteration(
            initial_policy,
            initial_values,
            environment,
            maxiter=MAXITER,
            tolerance=THRESHOLD,
            discount_factor=DISCOUNT_FACTOR
    )

    pprint(policy)
    pprint(values)

    with open(os.path.join(output_folder, 'policy.pkl'), 'wb') as f:
        pickle.dump(policy, f)

    with open(os.path.join(output_folder, 'values.pkl'), 'wb') as f:
        pickle.dump(values, f)


