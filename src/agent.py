import copy
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 
from src.env import Environment
from src.buffer import Buffer, Transition
class Agent():
    """reinforcement learning agent."""

    def __init__(self, env: Environment = None, gamma = 0.9, lr = 0.05, batch_size = 256):
        """
        Args:
            env: the environment.
            gamma: th discount factor for future rewards.
            lr: the learning rate.
        """
        self.buffer = Buffer(10000)
        self.env = env
        self.gama = gamma
        self.batch_size = batch_size
        input_size = env.get_state_size()
        output_size = env.get_output_size()
        hidden_layer_size = int(np.sqrt(input_size*output_size))
        input_layer = keras.layers.InputLayer(shape = (input_size,1))
        hidden_layer = keras.layers.Dense(units = hidden_layer_size, activation="relu")
        output_layer = keras.layers.Dense(units = output_size, activation="relu")
        self.model = keras.Sequential(layers=[input_layer, hidden_layer, output_layer])
        self.target_model = keras.Sequential(layers=[copy.deepcopy(input_layer), copy.deepcopy(hidden_layer), copy.deepcopy(output_layer)])

    def generate_experience(self, n_episodes: int, n_steps: int, random_chance = 0.9, decay_rate = 0.99, train = True):
        
        for episode in range(n_episodes):  
            for step in range(n_steps):
                random_chance *= decay_rate
                if np.random.choice([True, False]):
                    self.env.generate_blank_state()
                else:
                    self.env.generate_random_state()
                current_state = self.env.get_model_input()
                action = self.generate_action(random_chance = random_chance)

                reward = self.env.step(action)

                next_state = self.env.get_model_input()
                self.buffer.push(Transition(current_state, np.argmax(action), reward, next_state))
                if train and (len(self.buffer) >= self.batch_size):
                    self.optimise_weights()


    def generate_action(self, state: np.array, random_chance: float) -> np.array:
        if np.random.random() < random_chance:
            return np.random.random(size = self.env.get_output_size())
        else:
            return self.model.predict([state])
        
    def optimise_weights(self):
        batch = np.random.choice(self.buffer.memory,self.batch_size)
        states = batch[:, 0]
        actions = batch[:, 1]
        rewards = batch[:, 2]
        next_states = batch[:, 3]

