class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x*y
    
    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy
    
if __name__ == "__main__":
    apple = 100
    apple_num = 2
    tax = 1.1

    dprice = 1

    # 계층 정의
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # Forward
    apple_price = mul_apple_layer.forward(apple, apple_num) # apple_price = 200
    price = mul_tax_layer.forward(apple_price, tax) # price = 220

    # Backward
    dapple_price, dtax = mul_tax_layer.backward(dprice) # dtax = 200
    dapple, dapple_num = mul_apple_layer.backward(dapple_price) # dapple = 2.2, dapple_num = 110