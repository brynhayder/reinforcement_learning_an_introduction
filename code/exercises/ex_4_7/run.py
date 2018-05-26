#!/usr/bin/env python
"""
--------------------------------
project: code
created: 25/05/2018 23:12
---------------------------------

"""

from exercises.ex_4_7.jacks_car_rental.model import (
    Environment,
    Distributions,
    policy_iteration,
    initial_policy_values
)

CARS_MOVED_FREE = 1
STORAGE_COST = 4
STORAGE_THRESHOLD = 10


class AlteredEnvironment(Environment):
    def expected_reward(self, state, action):
        x, y = state
        rental_reward = self.rental_reward * (
                self.expected_rentals(x - action, self.distributions.x_rental)
                + self.expected_rentals(y + action, self.distributions.y_rental)
        )

        overnight_charge = STORAGE_COST if x - action > STORAGE_THRESHOLD or y + action > STORAGE_THRESHOLD else 0.

        return rental_reward - self.movement_cost * max(abs(action) - CARS_MOVED_FREE, 0) - overnight_charge


if __name__ == "__main__":
    import os
    import pickle
    import time

    from exercises.ex_4_7 import output_folder

    MAX_CARS = 20
    MAX_CAR_MOVES = 5
    DISCOUNT_FACTOR = 0.9
    THRESHOLD = 1e-2
    MAXITER = 10**4

    RENTAL_REWARD = 10
    MOVEMENT_COST = 2

    print("Building environment")
    start = time.time()
    environment = AlteredEnvironment(
            distributions=Distributions(),
            max_cars=MAX_CARS,
            max_car_moves=MAX_CAR_MOVES,
            rental_reward=RENTAL_REWARD,
            movement_cost=MOVEMENT_COST,
            populate_dynamics=True
    )

    print("Building environment took {:.2f}".format(time.time() - start))

    # with open(os.path.join(output_folder, f'environment_max_cars_{MAX_CARS}_max_moves_{MAX_CAR_MOVES}.pkl'), 'wb') as f:
    #     pickle.dump(environment, f)

    initial_policy, initial_values = initial_policy_values(environment)

    policy, values = policy_iteration(
            initial_policy,
            initial_values,
            environment,
            maxiter=MAXITER,
            tolerance=THRESHOLD,
            discount_factor=DISCOUNT_FACTOR
    )

    print(policy)
    print(values)

    with open(os.path.join(output_folder, 'policy.pkl'), 'wb') as f:
        pickle.dump(policy, f)

    with open(os.path.join(output_folder, 'values.pkl'), 'wb') as f:
        pickle.dump(values, f)
