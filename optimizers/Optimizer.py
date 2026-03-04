from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    @abstractmethod
    def step(self):
        pass