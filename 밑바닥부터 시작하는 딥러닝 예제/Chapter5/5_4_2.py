class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x+y
    
    def backward(self, dout):
        dx = dout # (= dout * 1)
        dy = dout # (= dout * 1)
        return dx, dy