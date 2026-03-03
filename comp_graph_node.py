import random

class ComputationGraphNode:
    def __init__(self,data,_children=(),_op = '',label = ''):
        if data is None: # NOTE assumes it to be weight 
            data = random.uniform(-1.0, 1.0)
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        out = ComputationGraphNode(self.data + other.data,(self, other),'+',f"{self.label} + {other.label}")
        def _backward(): 
            # notice this helps to have reference during call without passing them
            self.grad += 1.0*out.grad # as d(out) / d(self) = 1.0  
            other.grad += 1.0*out.grad # d(l)/d(out){->[out.grad]} * d(out)/d(other) 
            # it just follows chain rule     
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        out = ComputationGraphNode(self.data - other.data,(self, other),'-',f"{self.label} - {other.label}")
        def _backward(): 
            self.grad += 1.0*out.grad 
            other.grad += 1.0*out.grad    
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = ComputationGraphNode(self.data * other.data,(self, other),'*',f"{self.label} * {other.label}")
        def _backward():
            self.grad += other.data*out.grad # as d(out) / d(self) = other.data
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def dfs_topo_backward(self):
        topo :list[ComputationGraphNode] = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_topo(prev)
                topo.append(node)
            return
        build_topo(self)
        self.grad = 1
        for n in reversed(topo):
            n._backward()

        return
    
    def __repr__(self):
        return f"{self.label} - {self.data}"