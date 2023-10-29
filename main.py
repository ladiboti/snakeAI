import pygame as pg
import numpy as np
from random import randrange

# Constants
WINDOW = 1000
TILE_SIZE = 50
RANGE = (TILE_SIZE // 2, WINDOW - TILE_SIZE // 2, TILE_SIZE)
get_random_position = lambda: [randrange(*RANGE), randrange(*RANGE)]

# Initialize the snake's initial state
snake = pg.Rect([0, 0, TILE_SIZE - 2, TILE_SIZE - 2])
snake.center = get_random_position()
length = 10
segments = [snake.copy()]

# Initialize the Pygame screen
screen = pg.display.set_mode([WINDOW] * 2)
snake_dir = (0, 0)
time, time_step = 0, 110

# Initialize the food's initial position
food = snake.copy()
food.center = get_random_position()

# Initialize the Pygame clock
clock = pg.time.Clock()

# Direction dictionary to prevent reversing the snake's direction, possibly overcomplicated
dirs = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}

# Main game loop
while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            exit()
        if event.type == pg.KEYDOWN:
            # Check key presses and update the snake's direction
            if event.key == pg.K_w and dirs[pg.K_w]:
                snake_dir = (0, -TILE_SIZE)
                dirs = {pg.K_w: 1, pg.K_s: 0, pg.K_a: 1, pg.K_d: 1}
            if event.key == pg.K_s and dirs[pg.K_s]:
                snake_dir = (0, TILE_SIZE)
                dirs = {pg.K_w: 0, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}
            if event.key == pg.K_a and dirs[pg.K_a]:
                snake_dir = (-TILE_SIZE, 0)
                dirs = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 0}
            if event.key == pg.K_d and dirs[pg.K_d]:
                snake_dir = (TILE_SIZE, 0)
                dirs = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 0, pg.K_d: 1}

    # Clear the screen
    screen.fill('black')

    # Check if the snake collides with itself
    self_eating = pg.Rect.collidelist(snake, segments[:-1]) != -1

    # Check if the snake hits the screen boundaries or itself
    if snake.left < 0 or snake.right > WINDOW or snake.top < 0 or snake.bottom > WINDOW or self_eating:
        # Reset the game by repositioning the snake and food
        snake.center, food.center = get_random_position(), get_random_position()
        length, snake_dir = 1, (0, 0)
        segments = [snake.copy()]

    # Check if the snake eats the food
    if snake.center == food.center:
        food.center = get_random_position()
        length += 1

    # Draw the food and snake segments
    pg.draw.rect(screen, 'red', food)
    [pg.draw.rect(screen, 'green', segment) for segment in segments]

    # Update the snake's position and segments
    time_now = pg.time.get_ticks()
    if time_now - time > time_step:
        time = time_now
        snake.move_ip(snake_dir)
        segments.append(snake.copy())
        segments = segments[-length:]

    # Update the display and limit the frame rate
    pg.display.flip()
    clock.tick(50)  # Reduced the frame rate to 50 FPS
