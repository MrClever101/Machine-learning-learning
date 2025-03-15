import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 

class Environment():

    def step(self, action: np.array):
        """Step function."""
        index = np.argmax(action)
        reward = 10- self.grid.flatten()[index] 
        self.time +=1
        if self.time == 4:
            self.time = 0
            self.grid -=1
            grid = self.grid.flatten()[index]
            self.grid.reshape((5,5))

        return reward


    def generate_random_state(self):
        self.grid = np.random.randint(0,10, (5,5))
        time = 0

    def generate_blank_state(self):
        self.grid = np.zeros((5,5))
        time = 0
    
    def get_model_input(self) -> np.array:
        """This method returns the model input.
        Returns:
            the state.
        """
        return list(self.grid.flatten())+ [self.time]
    def get_state_size(self) -> int:
        return 26
    
    def get_output_size(self) -> int:
        return 26