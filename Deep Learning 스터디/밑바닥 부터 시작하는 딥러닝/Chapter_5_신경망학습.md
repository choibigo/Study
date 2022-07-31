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

# 5.5 활성화 함수 계층 구현하기


## ReLu 계층
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbDxdMs%2FbtqAJhUyVEu%2F6xCtrkc6NUNH7J6cHH98m0%2Fimg.png)


![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbURUsW%2FbtqAK40NRNW%2FrlUqsXRGG5dkGFJViUVvQK%2Fimg.png)

- 입력 값이 0보다 크면 역전파는 상류의 값을 그대로 하류로 보낸다.
- 순전파가 0이하인 경우 하류로 신호를 보내지 않는다.
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FPgp1D%2FbtqAKhl4f1d%2FCU03m5gaYX0TsZyKxYcFAK%2Fimg.png)

```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx                

```
- mask는 True/False로 구성된 넘파이 배열이다.
- 순전파 입력이 0 이하인 경우 True 를 갖는다.
- backward에서 True 값인 경우 0으로 변경한다. (더이상 역전파의 의미가 없다)

## 5.5.2 sigmoid 계층
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbuXy9x%2FbtqALQgKeCz%2FREaaWJwaxHwamvvo9rBND0%2Fimg.png)

#### sigmoid 계층의 계산 그래프
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F1c21b130-41a9-11ea-bc96-9d2b4aed3aec%2Ffig-5-20.png)

#### 1단계
- ```/``` 노드 y = 1/x
- 상류 값을 제곱하고 -값을 붙인다.

#### 2단계
- ```+``` 노드는 값의 변화 없이 뒤로 보낸다.

#### 3단계
- ```exp``` 노드의 미분 값은 exp(x)이다.
- exp 노드의 입력값인 -x 를 넣어 뒤로 보낸다.

#### 4단계
- ```x``` 노드는 순전파 떄의 값을 서로 바꾸어 보낸다.

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F20781300-41a9-11ea-baa8-418dfae37ccf%2Ffig-5-21.png)
- sigmoid의 역전파 함수는 x오 y로만 나타낼 수 있다.
- 즉 중간 계산 과정을 모두 묶어 단순한 노드 하나로 대체 할 수 있다.
- 또한 결고를 정리하면 아래와 같이 sigmoid의 역전파는 y하나만으로 계산 될 수 있다.
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F2aef5d70-41a9-11ea-8248-4760a63b1878%2Fe-5.12.png)

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F25463ce0-41a9-11ea-8248-4760a63b1878%2Ffig-5-22.png)

```python
def sigmoid:
    def __init__(self):
        out = None

    def forward(self, x):
        out = 1 / ( 1 + np.exp(-x))

        return out

    def backward(self, dout):
        return dout * out * (1-out)
        # 순전파의 결과를 보관했다 사용한다.
```

# 5.6 Affine / Softmax 계층 구현

## 5.6.1 Affine 계층
- 신경망의 순전파 때 수행하는 행렬의 곱은 기하학에서 Affine Transformation 이라 한다.
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbBHb0k%2FbtqALvjUUVe%2FP4srUaWP5vSp7gAXCnNfh0%2Fimg.png)

- X * W 에서 x와 w를 미분하는 식이 각각 다른다.
- **행렬곱에 대한 미분 식 유도 필요**
- 행렬의 순전파 입력과 역전파 결과의 형상은 같다.(또한 같아야 한다.)

## 5.6.2 배치용 Affine 계층
- 데이터 N개를 묶어 순전파 하는 경우 
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbg7kNw%2FbtqAK4tisBq%2F8uKeKfZdqqU8KKaLCEXF4k%2Fimg.png)

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dw = np.dot(self.X.T, dout)
        db = np.sum(dout, axis=0)
```

## 5.6.3 Softmax-with-Loss 계층
- 출력층에서 사용하는 소프트맥스 함수는 입력값을 정규화 하여 출력한다.
- 소프트맥스 계층을 구현을 손실함수와 함께 구현한다.
- 10클래수를 분류하는 문제라면 Softmax 계층의 입력은 10 개가 된다.

#### Softmax-With-Loss 계산 그래프
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcTrT4B%2FbtqAI33FH4t%2FkG3MbBq7Bl1uLDrkMnWF61%2Fimg.png)
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzX4Z3%2FbtqALt0HJ6j%2FyHOuB9073HnOwhgxlDExkK%2Fimg.png)
- Softmax의 입력 (a1, a2, a3)를 정규화한 값 (y1, y2, y3)
- 정답 레이블(t1, t2, t3)
- Softmax의 역전파는 (y1 - t1, y2 - t2, y3 - t3) 이다.
- 신경망의 역전파에서는 이 차이인 오차가 뒤로 전달 된다.(앞 계층으로 전해 진다.)
- 신경망의 학습 목적은 신경망 출력(Softmax의 출력)이 정답 레이블과 가까워지도록 가중치 매개변수의 값을 조정하는 것이다.
- 그래서 신경망의 출력과 정답레이블의 차이를 앞 계층에 전달 해야 한다.
- (y1 - t1, y2 - t2, y3 - t3)는 바로 Softmax 출력과 정답 레이블의 차이로 오차를 있는 그대로 나타낸다.
```
- 교차 엔트로피를 사용하니 역전파가 (y1 - t1, y2 - t2, y3 - t3)로 딱 떨어진다.
- 이것은 우연이 아니라 교차 엔트로피오차라는 함수가 이렇게 설계된 것이다.
```
- 정답 레이블 (0,1,0) 이고 Softmax 결과가 (0.3, 0.2, 0.5)를 출력 할때 확률은 0.2라서 정확도가 매우 낮은 상태이다.
- 이때 Softmax 계층의 역전파는 (0.3, -0.8, 0.5)라는 오차를 전파 한다.
- 결과 함수의 크기는 "오차"의 크기이다.
- 정답레이블이 음수인 이유는 softmax 정답 레이블의 출력이 커질 수록 오차 함수의 결과는 적어지는 반비례 관계 이기 때문이다.

```python
class softmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실 함수
        self.y = None # softmax의 출력
        self.t = None # 정답 레이블(원-핫 벡터)
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        # 출력층 이기 때문에 오차 역전파의 시작이다. 
        # 그렇기 때문에 dout의 값을 default로 설정한다.

        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        # 역전파 떄는 전파하는 값을 배치의수로 나누어서 데이터 1개당 오차를 앞계층으로 전파 한다.
        return dx
```