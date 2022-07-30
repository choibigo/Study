# 5.1 계산 그래프

## 5.1.1 계산 그래프로 풀다.
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F6ca21e20-41a3-11ea-b40d-6705eaadcebd%2Ffig-5-2.png)

```
슈퍼에서 1개에 100원인 사과 2개를 샀을때, 지불해야할 금액, 소비세가 10% 부과 된다.
```

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F848058e0-41a3-11ea-b40d-6705eaadcebd%2Ffig-5-3.png)

```
슈퍼에서 사과 2개, 귤 3개를 샀을떄 사과는 100원, 귤은 150이다. 소비세가 10%일때 지불 금액을 구하라
```
1) 그래프를 구성한다.
2) 그래프에서 계산을 왼쪽에서 오른쪽으로 한다.
- **2 단계**를 순전파 라고 한다.
- 오른쪽에서 왼쪽으로 진행하는 과정을 역전파 라고 한다.

## 5.1.2 국소적 계산
- 각 노드는 자신과 관련된 계산 외에는 다른 연산에는 신경 쓰지 않아도 된다.
- 전체 계산이 아무리 복잡하더라도 각 단계 에서 하는 일은 해당 노드의 **국소적 계산** 이다.


## 5.1.3 왜 계산 그래프로 푸는가?
- 전체가 아무리 보갖ㅂ해도 각 노드에서는 단순한 계산에 집중하여 문제를 단순화할 수 있다.
- 중간 계산 결과를 모두 보관할 수 있다.
- 역전파를 통해 **미분**을 효율적으로 계산할 수 있는 점에 있다.
- ```사과 가격이 최종 금액에 미치는 영향```을 알고 싶다.
- 이때 사과 가격에 대한 지불 금액의 미분을 구하는 문제와 같다.

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2Fdb6426f0-41a3-11ea-9e70-43b4cf1f0bf4%2Ffig-5-5.png)
- 지불 금액 = 사과의 개수 X 사과 금액 X 소비세 
- (지불 금액)' = 사과 금액 X 소비세 = > 2.2

## 5.2 연쇄법칙
- 역전파는 오른쪽에서 왼쪽으로 값을 전달 한다.
- **연쇄법칙** 원리를 이용한다.

## 5.2.1 계산 그래프의 역전파
- 순방향과 반대 방향으로 국소적 미분을 곱한다.

## 5.2.2 연쇄법칙이란?
- 합성함수의 미분은 함성함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.

#### 합성 함수
- 합성 함수란 여러 함수로 구성된 함수
- z = t^2
- t = x + y
- z = (x+y)^2

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F71a87990-41a4-11ea-b40d-6705eaadcebd%2Fe-5.4.png)


## 5.2.3 연쇄법칙과 계산 그래프
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2Fe47fd900-41a3-11ea-b40d-6705eaadcebd%2Ffig-5-7.png)
- 순전파와 반대 방향으로 국소적 미분을 곱하여 전달 한다. 
- x에 대한 z의 미분이 된다.


# 5.3 역전파

## 덧셈 노드의 역전파
- 덤샘 역전파의 경우 입력된 값을 그대로 보낸다.
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F55494a90-41a4-11ea-b40d-6705eaadcebd%2Ffig-5-9.png)
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2Faa4c23a0-41a4-11ea-bed1-737062fffe57%2Ffig-5-10.png)


## 곱셈 노드의 역전파
- 상류의 값에 순전파 떄의 입력 신호들을 서로 바꾼 값을 곱해서 보낸다.
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F1be4a500-41a5-11ea-8248-4760a63b1878%2Ffig-5-12.png)


## 사과 쇼핑의 예
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2Fdbf2e780-41a5-11ea-8248-4760a63b1878%2Ffig-5-14.png)


# 5.4 단순한 계층 구현하기

# 5.4.1 곱셈 계층
- 모든 계층은 forward()와 backward() 라는 공통 메서드를 갖는다.
```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y # 곱셈의 오차역전은 두 계수를 바꾼다.
        dy = dout * self.x

        return dx, dy
```

#### 사과2개 구입 문제 구현
```python
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price) # 220

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice) 
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
```
- 역전파 호출은 순전파의 반대이다.
- backward()가 받은 인수는 "순전파의 출력에 대한 미분"이다.

## 5.4.2 덧셈 계층
```python
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout # 상류에서 내려온 미분(dout)을 그대로 하류로 흘린다.
        dy = dout
    
        return dx, dy
```
