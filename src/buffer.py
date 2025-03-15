import numpy
import pickle
from collections import namedtuple, deque
Transition = namedtuple("transition", ["state", "action", "reward", "next_state"])
class Buffer():
    def __init__(self, capacity: int = 10000) -> None:
        self.memory = deque([], capacity)

    def push(self, t: Transition) -> None:
        """This method pushes the transition to the double-ended queue.
        Args:
            t: the transition to push.
        """
        self.memory.append(t)

    def save(self, name: str) -> None:
        """This method saves the buffer as a pickle file.
        Args:
            name: the name of the file to save to."""
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, "wb") as file:
            pickle.dump(self.memory, file)

    def load(self, name: str) -> None:
        """This method loads the buffer from a pickle file.
        Args:
            name: the name of the file to load."""
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, "rb") as file:
            self.memory = file.read()

    def __len__(self):
        return len(self.memory)
    