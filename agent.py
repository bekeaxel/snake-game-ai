import torch 
import random
import numpy as np
from collections import deque
from game import SnakeGame, Point, Snake, Apple
from model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BACTH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        snake = game.get_snake()
        apple = game.get_apple()
        head = snake.head()
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = snake.dir == (-1, 0)
        dir_r = snake.dir == (1, 0)
        dir_u = snake.dir == (0, -1)
        dir_d = snake.dir == (0, 1)

        state = [
            # danger straight
            (dir_l and snake.is_collision(point_l)) or
            (dir_r and snake.is_collision(point_r)) or
            (dir_u and snake.is_collision(point_u)) or
            (dir_d and snake.is_collision(point_d)),

            # danger right
            (dir_u and snake.is_collision(point_r)) or
            (dir_r and snake.is_collision(point_d)) or
            (dir_d and snake.is_collision(point_l)) or
            (dir_l and snake.is_collision(point_u)),

            # danger left
            (dir_l and snake.is_collision(point_d)) or
            (dir_r and snake.is_collision(point_u)) or
            (dir_u and snake.is_collision(point_l)) or
            (dir_d and snake.is_collision(point_r)),

            # direction of snake
            dir_l, 
            dir_r,
            dir_u,
            dir_d,

            # food location
            apple.pos.x < head.x,   # apple left
            apple.pos.x > head.x,   # apple right
            apple.pos.y < head.y,   # apple up
            apple.pos.y > head.y    # apple down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # train model after done on the memory for more training, overfitted?
    def train_long_memory(self):
        if len(self.memory) > BACTH_SIZE:
            mini_sample = random.sample(self.memory, BACTH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # let model train during playing, updating weights in nn after every move
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        if random.randint(0,200) < self.epsilon:
            final_move = [1,0,0]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        #get old state
        old_state = agent.get_state(game)

        #get move 
        move = agent.get_action(old_state)

        # perform move and get new state
        reward, done, score = game.play_step(move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, move, reward, new_state, done)

        #remember
        agent.remember(old_state, move, reward, new_state, done)

        if done:
            #train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.n_games, ' Score: ', ' Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()