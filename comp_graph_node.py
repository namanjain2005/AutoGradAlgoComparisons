class ComputationGraphNode:
    def __init__(self,data,_children=(),_op = '',label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        out = ComputationGraphNode(self.data + other.data,(self, other),'+')
        def _backward(): 
            # notice this helps to have reference during call without passing them
            self.grad += 1.0*out.grad # as d(out) / d(self) = 1.0  
            other.grad += 1.0*out.grad # d(l)/d(out){->[out.grad]} * d(out)/d(other) 
            # it just follows chain rule     
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = ComputationGraphNode(self.data * other.data,(self, other),'*')
        def _backward():
            self.grad += other.data*out.grad # as d(out) / d(self) = other.data
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        queue:list[ComputationGraphNode] = [self]
        while len(queue):
            start = queue[0]
            start._backward()
            queue.pop(0)
            for prev in start._prev:
                queue.append(prev)    
        return
    
    def __repr__(self):
        return f"{self.label} - {self.data}"