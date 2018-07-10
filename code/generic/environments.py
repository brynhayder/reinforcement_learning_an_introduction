#!/usr/bin/env python
"""
--------------------------------
project: code
created: 10/07/2018 15:52
---------------------------------

"""


class Gridworld:
    """Gridworld environment with walls. Suitable for building mazes."""
    def __init__(self, grid, start_position, goal_position, block_flag=1):
        self.grid = grid
        self.start_position = start_position
        self.goal_position = goal_position
        self.block_flag = block_flag

        self.current_state = self.start_position

    def reset(self):
        self.current_state = self.start_position
        return None

    def step(self, action):
        dx, dy = action
        x, y = self.current_state
        new_position = x + dx, y + dy
        if not self._on_grid(new_position) or self.grid[new_position] == self.block_flag:
            new_position = self.current_state
        self.current_state = new_position
        return new_position, self._reward(new_position), self._done(new_position)

    def _done(self, position):
        return position == self.goal_position

    def _reward(self, position):
        return -1. if position != self.goal_position else 0.

    def _on_grid(self, position):
        x, y = position
        return 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]