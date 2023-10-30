import pygame as pg
import numpy as np
import gymnasium as gym
import random
from gymnasium import spaces
from typing import List, Tuple, Callable

# Constants
WINDOW: int = 1000
TILE_SIZE: int = 50
RANGE: Tuple[int, int, int] = (TILE_SIZE // 2, WINDOW - TILE_SIZE // 2, TILE_SIZE)
get_random_position: Callable[[], List[int]] = lambda: [random.randrange(*RANGE), random.randrange(*RANGE)]


def create_binary_vector() -> np.ndarray:
    random_index: int = random.randint(0, 3)
    binary_vector: np.ndarray = np.zeros(4)
    binary_vector[random_index] = 1
    return binary_vector


class BinaryActionSpace:
    def __init__(self):
        self.binary_space_size: int = 4

    def sample(self) -> np.ndarray:
        binary_vector = np.zeros(self.binary_space_size)
        random_index = random.randint(0, self.binary_space_size - 1)
        binary_vector[random_index] = 1
        return binary_vector


class SnakeEnv(gym.Env):
    def __init__(self) -> None:
        super(SnakeEnv, self).__init__()

        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=255,
            shape=(WINDOW, WINDOW),
            dtype=np.uint8
        )

        self.action_space: BinaryActionSpace = BinaryActionSpace()

        self.snake: pg.Rect = pg.Rect([0, 0, TILE_SIZE - 2, TILE_SIZE - 2])
        self.snake.center = get_random_position()
        self.length: int = 1
        self.segments: List[pg.Rect] = [self.snake.copy()]
        self.snake_dir: Tuple[int, int] = (0, 0)

        self.food: pg.Rect = self.snake.copy()
        self.food.center = get_random_position()

    def reset(self, **kwargs) -> None:
        self.snake.center = get_random_position()
        self.food.center = get_random_position()
        self.length: int = 1
        self.snake_dir: Tuple[int, int] = (0, 0)
        self.segments: List[pg.Rect] = [self.snake.copy()]

    def step(self, action: np.ndarray) -> None:
        if action[0] == 1:  # up
            self.snake_dir = (0, -TILE_SIZE)
        elif action[1] == 1:  # down
            self.snake_dir = (0, TILE_SIZE)
        elif action[2] == 1:  # left
            self.snake_dir = (-TILE_SIZE, 0)
        elif action[3] == 1:  # right
            self.snake_dir = (TILE_SIZE, 0)

        if self.snake.center == self.food.center:
            self.food.center = get_random_position()
            self.length += 1

        self.snake.move_ip(self.snake_dir)
        self.segments.append(self.snake.copy())
        self.segments = self.segments[-self.length:]

    def _get_observation(self) -> np.ndarray:
        screen: pg.Surface = pg.Surface((WINDOW, WINDOW))
        screen.fill((0, 0, 0))
        pg.draw.rect(screen, (255, 0, 0), self.food)
        [pg.draw.rect(screen, (0, 255, 0), segment) for segment in self.segments]
        observation: np.ndarray = pg.surfarray.pixels3d(screen)
        return np.transpose(observation, axes=(1, 0, 2))

    def render(self) -> None:
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


env: SnakeEnv = SnakeEnv()
pg.init()
screen: pg.Surface = pg.display.set_mode((WINDOW, WINDOW))

for _ in range(1000):
    action: np.ndarray = env.action_space.sample()
    env.step(action)
    env.render()
