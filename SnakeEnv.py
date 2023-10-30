import pygame as pg
import numpy as np
import gymnasium as gym
import random
from gymnasium import spaces
from typing import List, Tuple, Callable

from numpy import ndarray

# Constants
WINDOW: int = 1000
TILE_SIZE: int = 50
RANGE: Tuple[int, int, int] = (TILE_SIZE // 2, WINDOW - TILE_SIZE // 2, TILE_SIZE)
FPS: int = 10000
get_random_position: Callable[[], List[int]] = lambda: [random.randrange(*RANGE), random.randrange(*RANGE)]


class BinaryActionSpace:
    def __init__(self, size: int) -> None:
        # left, right, up, down, might consider hardcoding it instead
        self.binary_space_size: int = size

    def sample(self) -> np.ndarray:
        # create a binary vector of specified size and set one of the elements randomly to 1
        # example output: [0, 1, 0, 0]
        binary_vector: np.ndarray = np.zeros(self.binary_space_size)
        random_index: int = random.randint(0, self.binary_space_size - 1)
        binary_vector[random_index] = 1

        return binary_vector


class SnakeEnv(gym.Env):
    def __init__(self) -> None:
        super(SnakeEnv, self).__init__()

        # define the observation space, currently the snake is able to observe the whole screen
        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=255,
            shape=(WINDOW, WINDOW),
            dtype=np.uint8
        )

        # creating the action space using the BinaryActionSpace class, with a specified binary space size of 4
        self.action_space: BinaryActionSpace = BinaryActionSpace(4)

        # initialize the game parameters
        self.snake: pg.Rect = pg.Rect([0, 0, TILE_SIZE - 2, TILE_SIZE - 2])
        self.snake.center = get_random_position()
        self.length: int = 1
        self.segments: List[pg.Rect] = [self.snake.copy()]
        self.snake_dir: Tuple[int, int] = (0, 0)

        self.food: pg.Rect = self.snake.copy()
        self.food.center = get_random_position()

    def reset(self, **kwargs) -> None:
        # reset the game state by repositioning the snake and the food
        self.snake.center = get_random_position()
        self.food.center = get_random_position()
        self.length: int = 1
        self.snake_dir: Tuple[int, int] = (0, 0)
        self.segments: List[pg.Rect] = [self.snake.copy()]

    def step(self, action: np.ndarray) -> Tuple[ndarray, int, bool, int]:
        # process the input action and update the current game state
        new_dir: Tuple[int, int] = self.snake_dir

        if action[0] == 1:  # up
            new_dir = (0, -TILE_SIZE)
        elif action[1] == 1:  # down
            new_dir = (0, TILE_SIZE)
        elif action[2] == 1:  # left
            new_dir = (-TILE_SIZE, 0)
        elif action[3] == 1:  # right
            new_dir = (TILE_SIZE, 0)

        # prohibiting snake moving to the opposite direction
        if (new_dir[0], -new_dir[1]) != self.snake_dir and (-new_dir[0], new_dir[1]) != self.snake_dir:
            self.snake_dir = new_dir

        # prohibiting self collision or moving out of the screen
        body_collision: bool = pg.Rect.collidelist(self.snake, self.segments[:-1]) != -1
        if (self.snake.left < 0 or
                self.snake.right > WINDOW or
                self.snake.top < 0 or
                self.snake.bottom > WINDOW or body_collision):
            self.reset()

        # when the snake coincides with the food, move the food to a new position and increment the length of the snake
        if self.snake.center == self.food.center:
            self.food.center = get_random_position()
            self.length += 1

        # calculate reward by the Manhattan distance of the snake's head and the food
        new_snake: pg.Rect = self.snake.copy()
        new_snake.move_ip(self.snake_dir)

        old_dist = abs(self.snake.x - self.food.x) + abs(self.snake.y - self.food.y)
        new_dist = abs(new_snake.x - self.food.x) + abs(new_snake.y - self.food.y)

        reward = calculate_reward(old_dist, new_dist)

        # move the snake, append the current position to the list of segments
        # update the segments to retain the length of the snake
        self.snake.move_ip(self.snake_dir)
        self.segments.append(self.snake.copy())
        self.segments = self.segments[-self.length:]

        # TODO: termination logic still hasnt implemented
        done: bool = False

        return self._get_observation(), reward, done, 69420

    def render(self) -> None:
        # render the current game state from the previous rgb array observation
        clock: pg.time.Clock = pg.time.Clock()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                pg.quit()

        screen.fill((0, 0, 0))
        surface: pg.Surface = pg.surfarray.make_surface(
            np.transpose(self._get_observation(), axes=(1, 0, 2))
        )
        screen.blit(surface, (0, 0))
        pg.display.flip()

        clock.tick(FPS)

    def _get_observation(self) -> np.ndarray:
        # return the current game state in a rgb array
        screen: pg.Surface = pg.Surface((WINDOW, WINDOW))
        screen.fill((0, 0, 0))
        pg.draw.rect(screen, (255, 0, 0), self.food)
        [pg.draw.rect(screen, (0, 255, 0), segment) for segment in self.segments]
        observation: np.ndarray = pg.surfarray.pixels3d(screen)

        return np.transpose(observation, axes=(1, 0, 2))


def calculate_reward(old_dist, new_dist) -> int:
    # define the reward logic based on the changes of game state
    reward: int = 0
    if new_dist < old_dist:
        reward = 10
    elif new_dist > old_dist:
        reward = -10

    return reward


# set up the Snake environment, initialize the Pygame module, and create a screen for rendering the game.
env: SnakeEnv = SnakeEnv()
pg.init()
screen: pg.Surface = pg.display.set_mode((WINDOW, WINDOW))

# run the game loop for a specified number of iterations, sampling actions, taking steps, and rendering the environment.
while True:
    action: np.ndarray = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    env.render()

# TODO: implement a reward and a termination function
