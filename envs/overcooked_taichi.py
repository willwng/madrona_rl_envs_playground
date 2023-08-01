import taichi as ti
import numpy as np
from typing import List

ti.init(arch=ti.cpu)

MAX_NUM_INGREDIENTS = 3
NONE = 0
TOMATO = 1
ONION = 2
DISH = 3
SOUP = 4
ALL_INGREDIENTS = [ONION, TOMATO]

AIR = 0
POT = 1
COUNTER = 2
ONION_SOURCE = 3
TOMATO_SOURCE = 4
DISH_SOURCE = 5
SERVING = 6


@ti.func
def move_in_direction(point, direction, width):
    ans = 0
    if direction == Action.NORTH:
        ans = point - width
    if direction == Action.SOUTH:
        ans = point + width
    if direction == Action.EAST:
        ans = point + 1
    if direction == Action.WEST:
        ans = point - 1
    if direction == Action.STAY:
        ans = point
    return ans


class Action(object):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    STAY = 4
    INTERACT = 5
    ALL_ACTIONS = [NORTH, SOUTH, EAST, WEST, STAY, INTERACT]
    NUM_ACTIONS = len(ALL_ACTIONS)


ObjectState = ti.types.struct(name=int, num_onions=int, num_tomatoes=int, cooking_tick=int)
PlayerState = ti.types.struct(position=int, orientation=int, proposed_position=int, proposed_orientation=int, held_object=ObjectState)


@ti.data_oriented
class TaichiSimulator:

    def __init__(self,
                 num_worlds: int,
                 terrain: List[int],
                 height: int,
                 width: int,
                 num_players: int,
                 start_player_x: List[int],
                 start_player_y: List[int],
                 placement_in_pot_rew: int,
                 dish_pickup_rew: int,
                 soup_pickup_rew: int,
                 recipe_values: List[int],
                 recipe_times: List[int],
                 horizon: int):
        self.num_worlds = num_worlds

        self.recipe_values = ti.field(ti.i32, shape=(16))
        self.recipe_values.from_numpy(np.array(recipe_values, dtype=np.int32))
        self.recipe_times = ti.field(ti.i32, shape=(16))
        self.recipe_times.from_numpy(np.array(recipe_times, dtype=np.int32))

        self.height = height
        self.width = width
        self.size = height * width
        self.terrain_mtx = ti.field(ti.i32, shape=(self.size))
        self.terrain_mtx.from_numpy(np.array(terrain, dtype=np.int32))
        self.list_start_player_positions = list(zip(start_player_x, start_player_y))
        self.num_players = num_players
        self.start_player_positions = ti.field(ti.i32, shape=(self.num_players))
        self.start_player_positions.from_numpy(np.array([pos[1] * self.width + pos[0] for pos in self.list_start_player_positions], dtype=np.int32))

        self.horizon = horizon
        self.placement_in_pot_rew = placement_in_pot_rew
        self.dish_pickup_rew = dish_pickup_rew
        self.soup_pickup_rew = soup_pickup_rew

        # state
        self.objects = ObjectState.field()
        self.players = PlayerState.field()
        self.timestep = ti.field(dtype=ti.i32)

        self.observation = ti.Vector.field(n=5 * self.num_players + 16, dtype=ti.i32)
        self.actions = ti.field(dtype=ti.i32)
        self.rewards = ti.field(dtype=ti.i32)
        self.dones = ti.field(dtype=ti.i32)

        self.allocate_fields()
        self.setup_base_observation()

    def allocate_fields(self):
        ti.root.dense(ti.i, self.num_worlds).dense(ti.j, self.num_players).place(
            self.players
        )

        ti.root.dense(ti.i, self.num_worlds).dense(ti.j, self.size).place(
            self.objects
        )

        ti.root.dense(ti.i, self.num_players).dense(ti.j, self.num_worlds).dense(ti.k, self.size).place(
            self.observation
        )

        ti.root.dense(ti.i, self.num_players).dense(ti.j, self.num_worlds).place(
            self.actions,
            self.rewards
        )

        ti.root.dense(ti.i, self.num_worlds).place(self.timestep, self.dones)

    @ti.func
    def get_recipe(self, soup):
        return (MAX_NUM_INGREDIENTS + 1) * soup.num_onions + soup.num_tomatoes

    @ti.func
    def get_terrain(self, pos):
        return self.terrain_mtx[pos]

    @ti.func
    def get_time(self, soup):
        return self.recipe_times[self.get_recipe(soup)]

    @ti.func
    def is_cooking(self, soup):
        return soup.cooking_tick >= 0 and soup.cooking_tick < self.get_time(soup)

    @ti.func
    def is_ready(self, soup):
        return soup.cooking_tick >= 0 and soup.cooking_tick >= self.get_time(soup)

    @ti.func
    def observationSystem(self, w: int, idx: int):
        for pos in range(self.size):
            if (self.horizon - self.timestep[w] < 40):
                self.observation[idx, w, pos][5 * self.num_players + 15] = 1
            else:
                self.observation[idx, w, pos][5 * self.num_players + 15] = 0

            for to_reset in range(5 * self.num_players):
                self.observation[idx, w, pos][to_reset] = 0

            self.observation[idx, w, pos][5 * self.num_players + 6] = 0
            self.observation[idx, w, pos][5 * self.num_players + 7] = 0
            self.observation[idx, w, pos][5 * self.num_players + 8] = 0
            self.observation[idx, w, pos][5 * self.num_players + 9] = 0
            self.observation[idx, w, pos][5 * self.num_players + 10] = 0
            self.observation[idx, w, pos][5 * self.num_players + 11] = 0
            self.observation[idx, w, pos][5 * self.num_players + 12] = 0
            self.observation[idx, w, pos][5 * self.num_players + 13] = 0
            self.observation[idx, w, pos][5 * self.num_players + 14] = 0

            if self.objects[w, pos].name == NONE:
                continue

            if self.objects[w, pos].name == SOUP:
                if self.terrain_mtx[pos] == POT:
                    if self.objects[w, pos].cooking_tick < 0:
                        self.observation[idx, w, pos][5 * self.num_players + 6] = self.objects[w, pos].num_onions
                        self.observation[idx, w, pos][5 * self.num_players + 7] = self.objects[w, pos].num_tomatoes
                    else:
                        self.observation[idx, w, pos][5 * self.num_players + 8] = self.objects[w, pos].num_onions
                        self.observation[idx, w, pos][5 * self.num_players + 9] = self.objects[w, pos].num_tomatoes

                        recipe_time = self.recipe_times[(MAX_NUM_INGREDIENTS + 1) * self.objects[w, pos].num_onions + self.objects[w, pos].num_tomatoes]
                        self.observation[idx, w, pos][5 * self.num_players + 10] = recipe_time - self.objects[w, pos].cooking_tick
                        if self.objects[w, pos].cooking_tick >= 0 and self.objects[w, pos].cooking_tick >= recipe_time:
                            self.observation[idx, w, pos][5 * self.num_players + 11] = 1
                else:
                    self.observation[idx, w, pos][5 * self.num_players + 8] = self.objects[w, pos].num_onions
                    self.observation[idx, w, pos][5 * self.num_players + 9] = self.objects[w, pos].num_tomatoes
                    self.observation[idx, w, pos][5 * self.num_players + 10] = 0
                    self.observation[idx, w, pos][5 * self.num_players + 11] = 1
            elif self.objects[w, pos].name == DISH:
                self.observation[idx, w, pos][5 * self.num_players + 12] = 1
            elif self.objects[w, pos].name == ONION:
                self.observation[idx, w, pos][5 * self.num_players + 13] = 1
            elif self.objects[w, pos].name == TOMATO:
                self.observation[idx, w, pos][5 * self.num_players + 14] = 1

        other_i = 1
        for i in range(self.num_players):
            pos = self.players[w, i].position
            if i == idx:
                self.observation[idx, w, pos][0] = 1
                self.observation[idx, w, pos][self.num_players + self.players[w, i].orientation] = 1
            else:
                self.observation[idx, w, pos][other_i] = 1
                self.observation[idx, w, pos][self.num_players + 4 * other_i + self.players[w, i].orientation] = 1
                other_i += 1

            if self.players[w, i].held_object.name == SOUP:
                self.observation[idx, w, pos][5 * self.num_players + 8] = self.players[w, i].held_object.num_onions
                self.observation[idx, w, pos][5 * self.num_players + 9] = self.players[w, i].held_object.num_tomatoes
                self.observation[idx, w, pos][5 * self.num_players + 10] = 0
                self.observation[idx, w, pos][5 * self.num_players + 11] = 1
            elif self.players[w, i].held_object.name == DISH:
                self.observation[idx, w, pos][5 * self.num_players + 12] = 1
            elif self.players[w, i].held_object.name == ONION:
                self.observation[idx, w, pos][5 * self.num_players + 13] = 1
            elif self.players[w, i].held_object.name == TOMATO:
                self.observation[idx, w, pos][5 * self.num_players + 14] = 1


    @ti.kernel
    def setup_base_observation(self):
        for w in range(self.num_worlds):
            for idx in range(self.num_players):
                self.players[w, idx].position = self.start_player_positions[idx]
            for idx in range(self.num_players):
                for pos in range(self.size):
                    v = self.terrain_mtx[pos]
                    if v > AIR:
                        self.observation[idx, w, pos][v - 1 + 5 * self.num_players] = 1
                self.observationSystem(w, idx)

    @ti.func
    def deliver_soup(self, w:int, i:int):
        obj = self.players[w, i].held_object
        val = self.recipe_values[(MAX_NUM_INGREDIENTS + 1) * obj.num_onions + obj.num_tomatoes]
        self.players[w, i].held_object.name = NONE
        self.players[w, i].held_object.num_onions = 0
        self.players[w, i].held_object.num_tomatoes = 0
        self.players[w, i].held_object.cooking_tick = -1
        return val

    @ti.func
    def soup_to_be_cooked_at_location(self, w, i_pos):
        ans = False
        if self.objects[w, i_pos].name == NONE:
            ans = False
        else:
            obj = self.objects[w, i_pos]
            ans = (
                obj.name == SOUP
                and not self.is_cooking(obj)
                and not self.is_ready(obj)
                and obj.num_onions + obj.num_tomatoes > 0
            )
        return ans

    @ti.func
    def soup_ready_at_location(self, w, i_pos):
        return self.objects[w, i_pos].name != NONE and self.is_ready(self.objects[w, i_pos])

    @ti.func
    def resolve_interacts(self, w: int):
        rew = 0
        for i in range(self.num_players):
            self.rewards[i, w] = 0
            if self.actions[i, w] != Action.INTERACT:
                continue

            pos = self.players[w, i].position
            o = self.players[w, i].orientation

            i_pos = move_in_direction(pos, o, self.width)
            terrain_type = self.terrain_mtx[i_pos]
            if terrain_type == COUNTER:
                if self.players[w, i].held_object.name != NONE and self.objects[w, i_pos].name == NONE:
                    self.objects[w, i_pos] = self.players[w, i].held_object
                    self.players[w, i].held_object.name = NONE
                    self.players[w, i].held_object.num_onions = 0
                    self.players[w, i].held_object.num_tomatoes = 0
                    self.players[w, i].held_object.cooking_tick = -1
                elif self.players[w, i].held_object.name == NONE and self.objects[w, i_pos].name != NONE:
                    self.players[w, i].held_object = self.objects[w, i_pos]
                    self.objects[w, i_pos].name = NONE
                    self.objects[w, i_pos].num_onions = 0
                    self.objects[w, i_pos].num_tomatoes = 0
                    self.objects[w, i_pos].cooking_tick = -1
            elif terrain_type == ONION_SOURCE:
                if self.players[w, i].held_object.name == NONE:
                    self.players[w, i].held_object.name = ONION
            elif terrain_type == TOMATO_SOURCE:
                if self.players[w, i].held_object.name == NONE:
                    self.players[w, i].held_object.name = TOMATO
            elif terrain_type == DISH_SOURCE:
                if self.players[w, i].held_object.name == NONE:
                    self.players[w, i].held_object.name = DISH
            elif terrain_type == POT:
                if self.players[w, i].held_object.name == NONE:
                    if self.soup_to_be_cooked_at_location(w, i_pos):
                        self.objects[w, i_pos].cooking_tick = 0
                else:
                    if self.players[w, i].held_object.name == DISH and self.soup_ready_at_location(w, i_pos):
                        self.players[w, i].held_object = self.objects[w, i_pos]
                        self.objects[w, i_pos].name = NONE
                        self.objects[w, i_pos].num_onions = 0
                        self.objects[w, i_pos].num_tomatoes = 0
                        self.objects[w, i_pos].cooking_tick = -1
                        rew += self.soup_pickup_rew
                    elif self.players[w, i].held_object.name == ONION or self.players[w, i].held_object.name == TOMATO:
                        if self.objects[w, i_pos].name == NONE:
                            self.objects[w, i_pos].name = SOUP
                            self.objects[w, i_pos].num_onions = 0
                            self.objects[w, i_pos].num_tomatoes = 0
                            self.objects[w, i_pos].cooking_tick = -1

                        if (not(self.objects[w, i_pos].cooking_tick >= 0 or self.objects[w, i_pos].num_onions + self.objects[w, i_pos].num_tomatoes == MAX_NUM_INGREDIENTS)):
                            if self.players[w, i].held_object.name == ONION:
                                self.objects[w, i_pos].num_onions += 1
                            else:
                                self.objects[w, i_pos].num_tomatoes += 1
                            self.players[w, i].held_object.name = NONE
                            self.players[w, i].held_object.num_onions = 0
                            self.players[w, i].held_object.num_tomatoes = 0
                            self.players[w, i].held_object.cooking_tick = -1
                            rew += self.placement_in_pot_rew
            elif terrain_type == SERVING:
                if self.players[w, i].held_object.name == SOUP:
                    rew += self.deliver_soup(w, i)
        for i in range(self.num_players):
            self.rewards[i, w] = rew

    @ti.func
    def _handle_collisions(self, w):
        should_update_or = False
        for idx0 in range(self.num_players):
            for idx1 in range(idx0 + 1, self.num_players):
                p1_old, p2_old = self.players[w, idx0].position, self.players[w, idx1].position
                p1_new, p2_new = self.players[w, idx0].proposed_position, self.players[w, idx1].proposed_position
                if p1_new == p2_new or (p1_new == p2_old and p1_old == p2_new):
                    should_update_or = True
                    break
            if should_update_or:
                break
        if should_update_or:
            for i in range(self.num_players):
                self.players[w, i].position = self.players[w, i].position
                self.players[w, i].orientation = self.players[w, i].proposed_orientation
        else:
            for i in range(self.num_players):
                self.players[w, i].position = self.players[w, i].proposed_position
                self.players[w, i].orientation = self.players[w, i].proposed_orientation

    @ti.func
    def _move_if_direction(self, w, i):
        if self.actions[i, w] == Action.INTERACT:
            self.players[w, i].proposed_position = self.players[w, i].position
            self.players[w, i].proposed_orientation = self.players[w, i].orientation
        else:
            new_pos = move_in_direction(self.players[w, i].position, self.actions[i, w], self.width)
            new_orientation = self.players[w, i].orientation if self.actions[i, w] == Action.STAY else self.actions[i, w]
            self.players[w, i].proposed_position = self.players[w, i].position if self.get_terrain(new_pos) != AIR else new_pos
            self.players[w, i].proposed_orientation = new_orientation

    @ti.func
    def resolve_movement(self, w):
        for i in range(self.num_players):
            self._move_if_direction(w, i)
        self._handle_collisions(w)

    @ti.func
    def step_environment_effects(self, w):
        self.timestep[w] += 1
        for pos in range(self.size):
            if self.objects[w, pos].name == SOUP and self.is_cooking(self.objects[w, pos]):
                self.objects[w, pos].cooking_tick += 1

    @ti.func
    def reset_system(self, w):
        if self.timestep[w] >= self.horizon:
            self.dones[w] = True
            self.timestep[w] = 0
            for i in range(self.num_players):
                self.players[w, i].position = self.start_player_positions[i]
                self.players[w, i].orientation = Action.NORTH
                self.players[w, i].proposed_position = self.players[w, i].position
                self.players[w, i].proposed_orientation = self.players[w, i].orientation
                self.players[w, i].held_object.name = NONE
                self.players[w, i].held_object.num_onions = 0
                self.players[w, i].held_object.num_tomatoes = 0
                self.players[w, i].held_object.cooking_tick = -1
            for i in range(self.size):
                self.objects[w, i].name = NONE
                self.objects[w, i].num_onions = 0
                self.objects[w, i].num_tomatoes = 0
                self.objects[w, i].cooking_tick = -1

    @ti.kernel
    def step(self):
        for w in range(self.num_worlds):
            self.dones[w] = False
            self.resolve_interacts(w)
            self.resolve_movement(w)
            self.step_environment_effects(w)
            self.reset_system(w)
            for i in range(self.num_players):
                self.observationSystem(w, i)
