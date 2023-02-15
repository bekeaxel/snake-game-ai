import pygame
import sys 
import random
import numpy as np
from collections import namedtuple

WIN_DIM = (800, 600)
CUBE_SIZE = 20
GRID_SIZE = (int(WIN_DIM[0] / CUBE_SIZE), int(WIN_DIM[1] / CUBE_SIZE))
SPEED = 30

GREEN = (119, 217, 89)
BLACK = (0, 0, 0)
RED = (196, 53, 0)
WHITE = (255, 255, 255)

Point = namedtuple('Point', 'x, y')
    
class Snake:
    def __init__(self):
        self.pos = Point(random.randint(0, GRID_SIZE[0]), random.randint(0, GRID_SIZE[1]))
        self.dir = (1, 0)
        self.color = GREEN
        self.body = [Point(self.pos.x - 2, self.pos.y), Point(self.pos.x - 1, self.pos.y), Point(self.pos.x, self.pos.y)]
        self.score = 0

    def move(self, action):
        # [1,0,0] straight, [0,1,0] right, [0,0,1] left

        # ändra dir beroende på action
        clock_wise = [(1,0), (0, -1), (-1, 0), (0, 1)]
        idx = clock_wise.index(self.dir)

        if np.array_equal(action, [0,1,0]): # right
            new_idx = (idx + 1) % 4 
            self.dir = clock_wise[new_idx]
        elif np.array_equal(action, [0,0,1]): # left
            new_idx = (idx - 1) % 4
            self.dir = clock_wise[new_idx]
        # else same direction    

        self.grow()
        self.body.pop(0)
        
    def eat(self, apple):
        if self.head() == apple.pos:
            self.grow()
            return True
        return False

    def is_collision(self, head):
        collision = False
        for cube in self.body[:len(self.body) - 1]:
            if cube == head:
                collision = True        
        return self.is_outside(head) or collision
       
    def grow(self):
        self.body.insert(0, (Point(self.dir[0] + self.head().x, self.dir[1] + self.head().y)))
    
    def head(self):
        return self.body[len(self.body) - 1]

    def is_outside(self, cube):
        return cube.x < 0 or cube.x >= GRID_SIZE[0] or cube.y < 0 or cube.y >= GRID_SIZE[1]

class Apple:
    def __init__(self, snake):
        self._snake = snake
        self.move()

    def move(self):
        self.pos = Point(random.randint(0, GRID_SIZE[0] - 1), random.randint(0, GRID_SIZE[1] - 1)) 
        if self.pos in self._snake.body:
            self.move()

class SnakeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Snake')
        self._window = pygame.display.set_mode(WIN_DIM)
        self._font = pygame.font.SysFont("Comic Sans MS", 24)
        self._clock = pygame.time.Clock()
        pygame.display.update()
        self.reset()
    
    def reset(self):
        self._snake = Snake()
        self._apple = Apple(self._snake)
        self._score = 0
        self._frame_iteration = 0

    def distance_to_apple(self):
        return abs(self._snake.head().x - self._apple.pos.x) + abs(self._snake.head().y - self._apple.pos.y)

    def play_step(self, action):
        reward = -1
        game_over = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        distance_before = self.distance_to_apple()
        # 1. move and check distance to apple
        self._snake.move(action)
        if distance_before > self.distance_to_apple():
            reward = 1

        # 2. check if game over  
        if self._snake.is_collision(self._snake.head()) or self._frame_iteration > 100 * len(self._snake.body):
            print('Frame iteration: ', self._frame_iteration)
            reward = -10
            game_over = True
            return reward, game_over, self._score

        # 3. Place new food or update
        if self._snake.eat(self._apple):
            self._score += 1
            reward = 15
            self._snake.grow()
            self._apple.move()
            
        # 4. update ui and clock   
        self._update_ui()
        self._clock.tick(SPEED)
        self._frame_iteration += 1

        #return reward, game over and score
        return reward, game_over, self._score 

    def _update_ui(self):
        self._window.fill(BLACK)

        # draw snake
        for cube in self._snake.body:
            pygame.draw.rect(self._window, GREEN, (cube.x * CUBE_SIZE, cube.y * CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

        #eyes
        if self._snake.dir == (1,0) or self._snake.dir == (-1,0):
            pygame.draw.circle(self._window, WHITE, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE / 2, self._snake.head().y * CUBE_SIZE + CUBE_SIZE / 3), 4)
            pygame.draw.circle(self._window, WHITE, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE / 2, self._snake.head().y * CUBE_SIZE + CUBE_SIZE * 2 / 3), 4)
            pygame.draw.circle(self._window, BLACK, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE / 2, self._snake.head().y * CUBE_SIZE+ CUBE_SIZE / 3), 1)
            pygame.draw.circle(self._window, BLACK, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE / 2, self._snake.head().y * CUBE_SIZE + CUBE_SIZE * 2 / 3), 1)
        else:
            pygame.draw.circle(self._window, WHITE, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE / 3, self._snake.head().y * CUBE_SIZE + CUBE_SIZE / 2), 4)
            pygame.draw.circle(self._window, WHITE, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE * 2 / 3, self._snake.head().y * CUBE_SIZE + CUBE_SIZE / 2), 4)
            pygame.draw.circle(self._window, BLACK, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE / 3, self._snake.head().y * CUBE_SIZE + CUBE_SIZE / 2), 1)
            pygame.draw.circle(self._window, BLACK, (self._snake.head().x * CUBE_SIZE + CUBE_SIZE * 2 / 3, self._snake.head().y * CUBE_SIZE + CUBE_SIZE / 2), 1)
       
        # draw apple
        pygame.draw.rect(self._window, RED, (self._apple.pos.x * CUBE_SIZE, self._apple.pos.y * CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

        #draw grid
        for col in range(GRID_SIZE[0]):
            pygame.draw.line(self._window, BLACK, (col * CUBE_SIZE, 0), (col * CUBE_SIZE, WIN_DIM[1]))        

        for row in range(GRID_SIZE[1]):
            pygame.draw.line(self._window, BLACK, (0, row * CUBE_SIZE), (WIN_DIM[0], row * CUBE_SIZE)) 

        pygame.draw.line(self._window, BLACK, (WIN_DIM[0] - 1, 0), (WIN_DIM[0] - 1, WIN_DIM[1]))
        pygame.draw.line(self._window, BLACK, (0, WIN_DIM[1] - 1), (WIN_DIM[0], WIN_DIM[1] - 1))

        #draw score
        score_label = self._font.render("Score: " + str(self._score) , 1, (255, 255, 255))
        self._window.blit(score_label, (WIN_DIM[0] - 120, 0))
        pygame.display.update()

    def get_apple(self):
        return self._apple
    
    def get_snake(self):
        return self._snake
