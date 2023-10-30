import pygame as pg
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import randrange

from gymnasium.spaces import MultiBinary

# Constants
WINDOW = 600
TILE_SIZE = 20
RANGE = (TILE_SIZE // 2, WINDOW - TILE_SIZE // 2, TILE_SIZE)
get_random_position = lambda: [randrange(*RANGE), randrange(*RANGE)]


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=255, shape=(WINDOW, WINDOW, 3), dtype=np.uint8)

        # Define action space
        self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Initialize game parameters
        self.snake = pg.Rect([0, 0, TILE_SIZE - 2, TILE_SIZE - 2])
        self.snake.center = get_random_position()
        self.length = 1
        self.segments = [self.snake.copy()]
        self.snake_dir = (0, 0)
        self.food = self.snake.copy()
        self.food.center = get_random_position()
        self.time, self.time_step = 0, 110

    def reset(self):
        # Reset the game by repositioning the snake and the food
        self.snake.center, self.food.center = get_random_position(), get_random_position()
        self.length, self.snake_dir = 1, (0, 0)
        self.segments = [self.snake.copy()]
        return self._get_observation()

    def step(self, action):
        # Process action and update game state
        new_dir = self.snake_dir
        done = False

        if action == 0:  # up
            new_dir = (0, -TILE_SIZE)
        elif action == 1:  # down
            new_dir = (0, TILE_SIZE)
        elif action == 2:  # left
            new_dir = (-TILE_SIZE, 0)
        elif action == 3:  # right
            new_dir = (TILE_SIZE, 0)

        # Check if the new direction is valid
        if (new_dir[0], -new_dir[1]) != self.snake_dir:
            self.snake_dir = new_dir

        self.snake.move_ip(self.snake_dir)
        self.segments.append(self.snake.copy())
        self.segments = self.segments[-self.length:]

        # Check for self-collision
        head = self.snake.copy()
        head.move_ip(self.snake_dir)

        if head.collidelist(self.segments[:-1]) is not None:
            self.reset()

        # Check for collision with walls
        if self.snake.left < 0 or self.snake.right > WINDOW or self.snake.top < 0 or self.snake.bottom > WINDOW:
            self.reset()

        if self.snake.center == self.food.center:
            self.food.center = get_random_position()
            self.length += 1
            print("food found")

        if self.length == 10:
            done = True

        # Calculate the reward
        old_distance = abs(self.snake.x - self.food.x) + abs(self.snake.y - self.food.y)
        new_distance = abs(head.x - self.food.x) + abs(head.y - self.food.y)
        reward = self._calculate_reward(self.length, old_distance, new_distance)

        return self._get_observation(), reward, done, {}

    def _calculate_reward(self, snake_length, old_distance, new_distance):
        # Define the reward logic based on the changes in snake length and distance to the food
        if new_distance < old_distance:  # Snake approaches the food
            reward = 10
        elif new_distance > old_distance:  # Snake moves away from the food
            reward = -10
        elif snake_length > 1:  # Snake grows by eating the food
            reward = 5
        else:  # Snake does not grow or move closer to the food
            reward = -1
        return reward

    def _get_observation(self):
        # Return the game state as the observation
        screen = pg.Surface((WINDOW, WINDOW))
        screen.fill((0, 0, 0))
        pg.draw.rect(screen, (255, 0, 0), self.food)
        [pg.draw.rect(screen, (0, 255, 0), segment) for segment in self.segments]
        observation = pg.surfarray.pixels3d(screen)
        return np.transpose(observation, axes=(1, 0, 2))

    def render(self):
        clock = pg.time.Clock()
        keys = pg.key.get_pressed()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                pg.quit()

        screen.fill((0, 0, 0))
        surface = pg.surfarray.make_surface(np.transpose(self._get_observation(), axes=(1, 0, 2)))
        screen.blit(surface, (0, 0))
        pg.display.flip()

        # clock.tick(100)


env = SnakeEnv()
observation = env.reset()

pg.init()
screen = pg.display.set_mode((WINDOW, WINDOW))


for i in range(10000):
    action = env.action_space.sample()  # NN.forward(observation)
    observation, reward, done, none = env.step(action)
    # (observation, reward, done)
    env.render()
    if done:
        print(f"Finished after {i + 1} steps")


observation_space = spaces.Tuple((
    spaces.Discrete(4),
))

selected_index = observation_space[0].sample()
action_vector = np.zeros(4)
action_vector[selected_index] = 1

print(action_vector)