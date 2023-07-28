# 밑바닥부터 시작하는 딥러닝 - 2강

## 퍼셉트론 이란
- 퍼셉트론은 다수의 신호를 입력으로 받아 하나의 신호를 출력합니다.

- ![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F99BDCE4D5B98A1022C95CF)

- X1과 X2는 입력신호, Y는 출력 신호 , W1과 W2는 가중치를 뜻한다.
- 입력 신호가 뉴런에 보내질 떄는 각각 고유한 가중치가 곱해진다.
- 뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력한다.(뉴런이 활성화 한다.)

## 2.2.1 AND 게이트
- AND 게이트를 퍼셉트론으로 표현할 때 W1, W2, 임계값을 정해야 한다.
- ex) W1, W2, 임계값 = (0.5, 0.5, 0.7)  

## 2.2.2 NAND 게이트
- ex) W1, W2, 임계값 = (-0.5, -0.5, -0.7) 
- AND 게이트를 구현하는 매개변수의 부호를 모두 반전하면 NAND게이트가 된다.

## 2.3.1 간단한 구현
```python
def AND(x1, x2):
    w1 = 0.5
    w2 = 0.5
    threshold = 0.7

    result = x1*w1 + x2*w2
    if result > threshold:
        return 1
    else:
        return 0
# AND(0,0) = 0
# AND(0,1) = 0
# AND(1,0) = 0
# AND(1,1) = 1
```

## 2.3.2 가중치와 편향 도입
- `b`를 편향이라 하며 퍼셉트론 입력신호에 가중치를 곱한 값과 편향을 합하여 그 값이 0을 넘으면 1을 출력 그렇지 않으면 0 을 출력 한다.
``` python
def AND_bias(x1, x2):
    input = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.7

    result = sum(input * weight) + bias

    if result > 0:
        return 1
    else:
        return 0
```
- threshold가 -b로 치환 되었다.
- 편향은 가중치 W1, W2와 기능이 다르다.
- 가중치는 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개 변수 이고, 편향은 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력) 하느냐를 조정하는 매개 변수 이다.

#### 편향은 왜 필요 할까?
![](https://media.geeksforgeeks.org/wp-content/uploads/Screenshot-from-2018-09-09-19-18-31.png)
![](https://media.geeksforgeeks.org/wp-content/uploads/Screenshot-from-2018-09-09-19-40-18.png)
- bias를 통해 activation function이 작동 할지 안할지를 관리 할 수 있다.
- 네트워크 학습시 더 유연하게 학습할 수 있다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCKOVF%2FbtrvAcbtWih%2FGBolgaIDOsaGVLKEkmKE21%2Fimg.png)
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc72Pzo%2FbtrvuOQW70k%2F6iWHHyLk85QNp0hG4IkoKK%2Fimg.png)
- bias가 있다면 좀더 데이터를 잘 나타낼 수 있다.
[참고](https://webnautes.tistory.com/1655)

## 퍼셉트론의 한계
- 지금까지 구현한 퍼셉트론으로 XOR 게이트를 구현할 수 없다.
- 지금까지 퍼셉트론은 직선으로 나눈 두영역을 만들기 때문에 XOR를 구현할 수 없다.

## 선형과 비선형
- 직선 영역으로 나누는 것이 **선형**
- 곡선 영역으로 나누는 것이 **비선형**

## 다층 퍼셉트론
- 퍼셉트론의 층을 쌓아 다층 퍼셉트론을 만들 수 있다.
- ![image](https://upload.wikimedia.org/wikipedia/commons/a/a2/254px_3gate_XOR.jpg)
- AND, NAND, OR 게이트를 사용해 XOR게이트를 만들 수 있다.
```python
def XOR(x1, x2):
    s1 = NAND_bias(x1, x2)
    s2 = OR_bias(x1, x2)
    result = AND_bias(s1, s2)

    return result
```
- ![image](https://velog.velcdn.com/post-images/dscwinterstudy/c7389f50-3a21-11ea-ac7b-8be67350a17c/2-13XOR%EC%9D%98-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.png)
- 이처럼 층이 여러개인 퍼셉트론을 다층 퍼셉트론 이라 한다.
  - 0층의 두 뉴런이 입력 신홀르 받아 1층 뉴런으로 신호를 보낸다.
  - 1층 뉴런이 2층의 뉴런으로 신호를 보내고, 2층 뉴련은 Y를 출력한다.
- 퍼셉트론은 층을 쌓아 더 다양한 것을 표현할 수 있다.