#!/usr/bin/env python
"""
--------------------------------
project: code
created: 09/07/2018 16:08
---------------------------------


I changed the example to give a negative expected_reward for each timestep because this means that I don't
have to implement random tie-breaking with max in the action-selection. It also makes convergence faster
since rewards are seen immediately.

"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from generic import policies, utils
from generic.agents.dyna_q import DynaQAgent, DynaQPlusAgent
from generic.environments import Gridworld

POSSIBLE_ACTIONS = [(0, 1), (1, 0), (-1, 0), (0, -1)]


def initial_action_values(grid_shape, possible_actions=POSSIBLE_ACTIONS, v=0.):
    return {
        s: {a: v for a in possible_actions} for s in np.ndindex(*grid_shape)
    }


def mark_trajectory(grid, states, agent_flag=2):
    g = grid.copy()
    for s in states:
        g[s] = agent_flag
    return g


def learning_curve(agent_maker, environment, n_steps, n_iters=10):
    args = [
        (agent_maker(environment.grid.shape), environment, n_steps, True)
        for _ in range(n_iters)
    ]
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(
                utils.run_continuous,
                *zip(*args)
        )
    all_rewards = np.c_[[np.array(r.rewards) for r in all_results]].T
    return np.cumsum(all_rewards + 1, axis=0)


def run_blocking_maze(agent_maker):
    grid = blocked_grid(open='left')
    agent = agent_maker(grid.shape)

    environment = Gridworld(grid=grid, start_position=(5, 3), goal_position=(0, 8))
    r1 = utils.run_continuous(agent, environment, 1000, True)

    environment.grid = blocked_grid(open='right')
    environment.reset()
    r2 = utils.run_continuous(agent, environment, 2000, True)
    return utils.Results(
            states=r1.states + r2.states,
            actions=r1.actions + r2.actions,
            rewards=r1.rewards + r2.rewards
    )


def blocking_maze_learning_curve(agent_maker, n_iters=10):
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(
                run_blocking_maze,
                [agent_maker for _ in range(n_iters)]
        )
    all_rewards = np.c_[[np.array(r.rewards) for r in all_results]].T
    return np.cumsum(all_rewards + 1, axis=0)


def blocked_grid(shape=(6, 9), block_row=3, open='left', block_flag=1):
    grid = np.zeros(shape)
    grid[block_row, :] = block_flag

    if open == 'left' or open == 'both':
        grid[block_row, 0] = 0
    if open == 'right' or open == "both":
        grid[block_row, -1] = 0

    return grid


if __name__ == "__main__":
    import os

    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    import plotting
    import constants as c


    def dyna_q_agent_maker(grid_shape):
        epsilon_greedy_policy = policies.EpsilonGreedyPolicy(
                action_values=initial_action_values(grid_shape),
                epsilon=0.1
        )
        return DynaQAgent(
                policy=epsilon_greedy_policy,
                alpha=0.1,
                gamma=0.95,
                n_plan_iter=50,
                random_state=np.random.RandomState(None)
        )

    def dyna_q_plus_agent_maker(grid_shape):
        greedy_policy = policies.GreedyPolicy(
                action_values=initial_action_values(grid_shape)
        )
        return DynaQPlusAgent(
                policy=greedy_policy,
                alpha=0.1,
                gamma=0.95,
                kappa=0.01,
                n_plan_iter=50,
                random_state=np.random.RandomState(None)
        )

    def altered_dyna_q_plus_agent_maker(grid_shape):
        policy = policies.TimeBiasedPolicy(
                action_values=initial_action_values(grid_shape=grid_shape),
                kappa=0.01
        )
        return DynaQAgent(
                policy=policy,
                alpha=0.1,
                gamma=0.95,
                n_plan_iter=50,
                random_state=np.random.RandomState(None)
        )


    dyna_q_curve = blocking_maze_learning_curve(dyna_q_agent_maker)
    dyna_q_plus_curve = blocking_maze_learning_curve(dyna_q_plus_agent_maker)
    altered_dyna_q_plus_curve = blocking_maze_learning_curve(altered_dyna_q_plus_agent_maker)

    with plt.rc_context(plotting.rc()):
        fig, ax = plt.subplots(1)
        ax.plot(np.mean(dyna_q_curve, axis=1), label="Dyna Q")
        ax.plot(np.mean(dyna_q_plus_curve, axis=1), label="Dyna Q Plus")
        ax.plot(np.mean(altered_dyna_q_plus_curve, axis=1), label="Altered Dyna Q Plus")
        ax.grid(alpha=0.1)
        ax.axvline(1000, color='k', zorder=-1, ls='--', label='Block')
        ax.legend()

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Blocking Maze Example", fontsize=14)

        plotting.savefig(
                fig,
                os.path.join(
                        c.Paths.output,
                        'ex_8_4',
                        'dyna_q_comparison.png'
                )
        )

    # ---------------------------
    # Comparison on a static maze
    # ---------------------------
    # grid = blocked_grid(open='left')
    # environment = Gridworld(grid=grid, start_position=(5, 3), goal_position=(0, 8))
    #
    # dynaq_cumulative_rewards = learning_curve(
    #         dyna_q_agent_maker,
    #         environment,
    #         n_steps=3000,
    #         n_iters=10
    # )
    #
    # dynaqplus_cumulative_rewards = learning_curve(
    #         dyna_q_plus_agent_maker,
    #         environment,
    #         n_steps=3000,
    #         n_iters=10
    # )
    #
    # plt.plot(np.mean(dynaq_cumulative_rewards, axis=1), label='Dyna-Q')
    # plt.plot(np.mean(dynaqplus_cumulative_rewards, axis=1), label='Dyna-Q+')
    # plt.legend()
    # print('ready')
    # plt.show()

    # ---------------
    # running a single example
    # ---------------
    # grid = blocked_grid(open='left')
    # environment = Gridworld(grid=grid, start_position=(5, 3), goal_position=(0, 8))
    # print(grid)
    #
    # a = altered_dyna_q_plus_agent_maker(grid.shape)
    # results = utils.run_continuous(a, environment, n_steps=3000)
    #
    # a.set_policy_type(policies.GreedyPolicy)
    # greedy_episode = utils.run_episode(a, environment, update=False)
    #
    # print('-------------------------------------------------')
    # print("Greedy Trajectory")
    # g = mark_trajectory(grid, greedy_episode.states, 2)
    # print(g)
    #
    # fig, ax = plt.subplots(1)
    # ax.imshow(g)
    # ax.set_title("Greedy Episode")
    # plt.show()
