#!/usr/bin/env python
"""
--------------------------------
project: code
created: 31/05/2018 16:52
---------------------------------

"""
import numpy as np
import matplotlib;

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Made the greedy policy not able to select 0 as an action to get a non-trivial policy


class Environment:
    def __init__(self, p_win, winning_capital=100):
        self.p_win = p_win
        self.p_lose = 1 - p_win
        self.winning_capital = winning_capital
        self.possible_states = range(1, winning_capital)

    def possible_actions(self, state):
        return range(min(state, self.winning_capital - state + 1))

    def single_value_update(self, action, state, values):
        out = self.p_win * values[state + action] + self.p_lose * values[state - action]
        # if state + action >= self.winning_capital:
        #     out += self.p_win
        return out


def value_update(state, values, environment):
    return np.max(
            [environment.single_value_update(a, state, values) for a in environment.possible_actions(state)]
    )


def value_iteration(values, environment, tolerance, verbose=False):
    delta = tolerance
    sweeps = 0
    vals = [values]
    while delta >= tolerance:
        old_values = values
        values = [value_update(s, old_values, environment) for s in environment.possible_states]
        values.append(1);
        values.insert(0, 0)
        values = np.array(values)
        delta = np.max(np.abs(old_values - values))
        sweeps += 1
        vals.append(values)
        if verbose:
            print(f"End of sweep {sweeps} delta = {delta}")
    return vals, greedy_policy(values, environment)


def action_values(state, values, environment):
    return [environment.single_value_update(a, state, values) for a in environment.possible_actions(state)]


def greedy_action(state, values, environment):
    avs = action_values(state, values, environment)
    avs[0] = 0
    return np.argmax(avs)


def greedy_policy(values, environment):
    return np.array([
        greedy_action(state, values, environment) for state in environment.possible_states
    ])


def initial_values(environment):
    values = np.zeros(len(environment.possible_states) + 2)
    values[0] = 0
    values[-1] = 1
    return values


if __name__ == "__main__":
    import os

    import constants as c
    from exercises import plotting


    def plot(values_list, policy, i, p_win, name=None, legend=True):
        with plt.rc_context(plotting.rc()):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            for sweep, v in enumerate(values_list, start=1):
                ax1.plot(v, label=sweep, lw=0.5)

            for ax in ax1, ax2:
                ax.grid(alpha=0.1, ls=':')

            if legend:
                ax1.legend(title="Sweep", bbox_to_anchor=(1, 1))
            ax1.set_title(
                    fr"Optimal Values: $\theta=10^{{{-i}}}$",
                    x=0.05,
                    y=0.95,
                    ha='left',
                    va='top',
                    fontsize=10
            )
            ax2.plot(policy)
            ax2.set_title(
                    fr"Optimal Policy: $\theta=10^{{{-i}}}$",
                    x=0.05,
                    y=0.95,
                    ha='left',
                    va='top',
                    fontsize=10
            )

            plt.suptitle(fr"$\mathbb{{P}}(\mathtt{{win}})={p_win}$")
            if name is not None:
                plt.savefig(
                        os.path.join(
                                c.Paths.exercise_output,
                                'ex_4_9',
                                name + '.eps'
                        ),
                        format='eps',
                        dpi=1000,
                        bbox_inches='tight'
                )

        return fig, (ax1, ax2)


    i = 3

    for p_win in [0.25, 0.55]:
        environment = Environment(
                p_win=p_win,
                winning_capital=100
        )

        values, policy = value_iteration(
                values=initial_values(environment),
                environment=environment,
                tolerance=10 ** -i,
                verbose=True
        )

        plot(
            values,
            policy,
            name='values_and_policy_pwin_{:2.0f}'.format(100 * p_win),
            i=i,
            p_win=p_win,
            legend=p_win < 0.5
        )

    # plt.show()
