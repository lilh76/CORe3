import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__)))[:-12])
from utils.debug import *
import numpy as np
import gym
from gym import Env
from gym.utils import seeding

from enum import Enum
from collections import namedtuple, defaultdict
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.field_size = None
        self.current_step = None

    def setup(self, position, field_size):
        self.position = position
        self.field_size = field_size

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"

class gridEnv(Env):
    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position"]
    )
    def __init__(self, field_size, max_food, max_episode_steps) -> None:
        self.reward_dim = 5
        self.seed()
        self.players = [Player()]
        self.field = np.zeros(field_size, np.int32)
        self.max_food = max_food
        self._food_spawned = 0.0
        self._game_over = None
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)]))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()]))
        self._max_episode_steps = max_episode_steps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
         

        max_food = self.max_food
        min_obs = [-1, -1]
        max_obs = [field_x-1,field_y-1]

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )
    
    def adjacent_food_location(self, row, col):
        if row > 0 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 0 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def spawn_player(self):
        for player in self.players:
            row = 0
            col = 0
            player.setup(
                (row, col),
                self.field_size,
            )

    def spawn_food(self, max_food):
         
        self.field[self.rows - 1, 0] = 1
        self.field[0, self.cols - 1] = 1
        self.field[self.rows - 1, self.cols - 1] = 1
        self._food_spawned = self.field.sum()
        return

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
         
         

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def _make_obs(self):

        return self.players[0].position
    
    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.field_exp = np.zeros(self.field_size, np.int32)
        self.spawn_player()

        self.spawn_food(self.max_food)
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()
        return self._make_obs()

    def step(self, _actions):
        actions = [_actions]
        reward = np.zeros(self.reward_dim)
        self.current_step += 1

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

         
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE
         
        collisions = defaultdict(list)

         
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)

        for k, v in collisions.items():
            if len(v) > 1:   
                continue
            v[0].position = k

        player = self.players[0]
        player_pos = player.position
        def manhattan_dist(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        max_dist = self.rows + self.cols - 2
        reward[0] = max_dist - manhattan_dist(player_pos, (self.rows-1,  0))
        reward[1] = max_dist - manhattan_dist(player_pos, (self.rows-1,  self.cols//2))
        reward[2] = max_dist - manhattan_dist(player_pos, (self.rows-1,  self.cols-1))
        reward[3] = max_dist - manhattan_dist(player_pos, (self.rows//2, self.cols-1))
        reward[4] = max_dist - manhattan_dist(player_pos, (0,            self.cols-1))

        self._game_over = (self.field.sum() < self._food_spawned or self._max_episode_steps <= self.current_step)
        self._gen_valid_moves()
        
        return self._make_obs(), reward, self._game_over, {}

    def close(self):
        return