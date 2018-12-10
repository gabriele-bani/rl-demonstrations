import random


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) == self.capacity:
            self.memory = self.memory[1:]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample
    
    def reset(self):
        del self.memory
        self.memory = []
    
    def __len__(self):
        return len(self.memory)