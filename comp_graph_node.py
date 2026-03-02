class ComputationGraphNode:
    def __init__(self,data,label):
        self.data = data
        self.grad = 0.0
        self.label = label
    
    def __add__(self, other):
        return ComputationGraphNode(self.data + other.data,'+')

    def __mul__(self, other):
        return ComputationGraphNode(self.data*other.data,'*')
    
    def __repr__(self):
        return f"{self.label} - {self.data}"