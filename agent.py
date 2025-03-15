import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 
from env import Environment
class Agent():
    """reinforcement learning agent."""

    def __init__(self, env: Environment = None, gamma = 0.9, lr = 0.05):
        """
        Args:
            env: the environment.
            gamma: th discount factor for future rewards.
            lr: the learning rate.
        """
        self.env = env
        input_size = env.get_state_size()
        output_size = env.get_output_size()
        hidden_layer_size = int(np.sqrt(input_size*output_size))
        input_layer = keras.layers.Input(shape=(input_size,))
        hidden_layer = keras.layers.Dense(units = hidden_layer_size)
        output_layer = keras.layers.Dense(units = output_size)
        self.model = keras.Sequential(layers=[input_layer, hidden_layer, output_layer])
        

    def generate_experience(self, n_episodes: int, n_steps: int, random_chance = 0.9, decay_rate = 0.99):
        
        for episode in n_episodes:  
            random_chance *= decay_rate
            if np.random.choice([True, False]):
                self.env.generate_blank_state()
            else:
                self.env.generate_random_state()
            current_state = self.env.get_model_input()
            action = self.generate_action(random_chance = random_chance)


    def generate_action(self, state: np.array, random_chance: float) -> np.array:
        if np.random.random() < random_chance:
            return np.random.random(size = self.env.get_output_size())
        else:
            return self.model.predict([state])