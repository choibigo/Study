# 학습 관련 기술들

# 6.1 매개변수 갱신
- 신경망 학습의 목적은 손실함수의 값을 가능한 낮추는 매개변수를 찾는 과정을 최적화 라고 한다.
- 매개변수의 기울기를 구해 기울어진 방향으로 매개변수 값을 갱신하는 과정을 확률적 경사 하강법(SGD)라 한다.

## 6.1.2 확률적 경사 하강법
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F72ae1bd0-417a-11ea-b254-2b6843f7eefa%2Fimage.png)
<br>
- 기존 가중치에 (손실함수의 기울기 X 학습률) 만큼 이동 한다.

```py
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

## 6.1.3 SGD의 단점
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMTUy/MDAxNTAxMTIxODE1NDM0.OqqNgMATDOxdpQwhB8xB45xtRMI3DrlbTBAVgM8auoQg.dPsvhWdAZmfA8H1BvvXnifZBIetbANydhH_xajC4U_0g.PNG.cjswo9207/e_6.2.png?type=w2)
<br>
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjU3/MDAxNTAxMTIxODU0MTYx.RZxiBW4kDiXReqK1UDxfBAffbM-pWJQwEI82m-evhucg.N_UhoBOXaCVYUPsP4wuoi4UUL1qQ_5iNEB000_QP1S4g.PNG.cjswo9207/fig_6-1.png?type=w2)
<br>
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMTMx/MDAxNTAxMTIxOTIwOTE4.b5qnZS_a5u0UOQV16EQzWZpGXpmAkjKeqNVE8COqaNQg.sO1Li2aqIyVhpv3h1NfwLCuyqUfw9G1GZR-Nbsy0trgg.PNG.cjswo9207/fig_6-2.png?type=w2)
- 기울기는 Y축으로 크고 X축으로 작다는것이 특징이다.
- Y축은 가파르고 X축 방향을 완만한 것이다. 
- 최소값 되는 장소는 (0,0)이지만, 기울기가 각 기울기는 0,0을 가리키지 않는다.

![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMTU0/MDAxNTAxMTIyMDAwNTA1.E7zIAUXugKPAA5WiBXpF7HtATQ8nVjfBvJF-LJwqtsIg.52gqDjo2sPPup4Bfz5hOtp_b1svFcFItoYAJreHPp6Ag.PNG.cjswo9207/fig_6-3.png?type=w2)
- Y축으로는 크게 움직이나 X축으로는 작게 움직이게 된다.
- 그렇기 때문에 심하게 굽어진 움직임을 보이게 된다.
- 즉, 기울기의 성격에 따라 탐색 경로가 비효율적으로 될 수 있다.
- 모멘텀, AdaGrad, Adam은 SGD의 단점을 개선해 주는 방법이다.

## 6.1.4 모멘텀
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMiAg/MDAxNTAxMTIyODA5NTQ3.-JX9XX-mWO-yDRF_iauTPMKJTBSQ0InC7FFAHIef3UMg.7jXFIOlhf8dh84ckF6Y3XR4rVv3q5YXo0gjFtBsrvX0g.PNG.cjswo9207/e_6.3.png?type=w2)
<br>
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMTE4/MDAxNTAxMTIyODA5NzI4.VYoxVLd0N-gbcfmJa7kng_ezd_Ljc-q_uRQA94OD0RAg.5zo-fvLQMHkF8P2YyfS_zlqskF_Koai7tSvXbgNUFgwg.PNG.cjswo9207/e_6.4.png?type=w2)
<br>

- v는 물리학에서 말하는 속도에 해당한다.
- 기울기 방향으로 힘을 받아 물체가 가속된다는 물리 법칙을 나타 낸다.
```py
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
```
- v 는 update()가 처음 호출될 때 매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장한다.

![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjQ5/MDAxNTAxMTIzNzcxMjY4.pW8LeX0QqcoQMI1qBQmpYf9skv8k0G2SBAqrmk_uP84g.CKPbYA_icOECAZXhBTB_g1qzmDRdtL3vUbEfbU4Y8yMg.PNG.cjswo9207/fig_6-5.png?type=w2)

- SGD에 비해서 "지그재그 정도"가 덜한다.
- x축의 힘은 작지만 방향은 변하지 않아서 한방향으로 일정하게 가속하고, y축의 힘은 크기만 방향이 변해 위아래로 상충하여 방향과 속도가 일정하지 않다.
- SGD보다 X축 방향으로 빠르게 수렴한다.

## 6.1.5 AdaGrad
- 학습률은 너무 작으면 수렴시간이 너무 오래 걸리고 학습률이 크면 발산하여 학습이 제대로 이뤄지지 않는다.
- 학습을 진행하면서 학습률을 점차 줄여가는 방법을 Learning Rate Decay가 있다. 
- 학습률감소를 발전 시킨 방법이 AdaGrad이다. 각 매개 변수에 맞게 학습률을 조절하는 방법이다.
<br>
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjc3/MDAxNTAxMTI0MTM3Mjc4.Fm_iYsJ5Vbb_QqRPNHZW1ES_lLGHvXQVN-k_WOkCzpIg.UZn0S1DonmiphH6ccxXU_HMXoh8RD58foaOtGVEUaxwg.PNG.cjswo9207/e_6.5.png?type=w2)
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMTAy/MDAxNTAxMTI0MTM3NTMy.-NU8zO3XRu5Bp2Zf7xvfHpFTUD8n_00hjAA3eNYa7kYg.36l9rO9irMqCa8xd5qLjZmOwyqB7qKP0Mmw4E2vU24Ug.PNG.cjswo9207/e_6.6.png?type=w2)
<br>
- h는 기존 기울기 값을 제곱하여 계속 더해 준다.
- 매개변수를 갱신할때 1 / sqrt(h) 를 곱해 학습률을 조정한다.
- 매개변수의 원소 중 많이 움직인 원소는 학습률이 낮아 지고, 덜 움직인 원소는 학습률이 커진다. (많이 움직였다 = 그전 기울기가 크다)
- 학습률이 매개변수의 원소마다 다르게 적용된다.

```
- AdaGrad는 학습을 계속할 수록 갱신 강도가 0이 되어 갱신을 하지 않는 문제가 있다.
- PMSProp는 과거 모든 기울기를 균일하게 더하는 것이 아니라 먼 과거의 기울기는 잊고 새로운 기울기 정보를 크게 반영해서 이를 해결한다.
```

```py
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr*grads[key] / (np.sqrut(self.h[key]) + le-7)
```

![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfNDQg/MDAxNTAxMTI0NTczMDQz.1HzCY4kj0AqKMon65oOJdYfaIxGveuUjawHvH_XOtnkg.0Pf9RCSvNNbZiEqybl7EOt2b49lmHFcWSbFeBpnQG2Yg.PNG.cjswo9207/fig_6-6.png?type=w2)
<br>
- Y축 방향이 기울기가 커서 처음에는 크게 움직이지만 그 움직에 비례해 갱신 정도도 큰폭으로 작아지도록 조정 된다.
- Y축 방향으로 갱신 정도가 빠르게 약해지고, 지그재그 움직임이 줄어 든다.

## 6.1.6 Adam
- 모멘텀은 기울기 방향, AdaGrad는 기울기 크기에 따라 다음 매개변수를 결정 했다. 
- 이 두 기법을 융합한 것이 Adam 이다.
- Adam은 하이퍼파라미터가 3개 이다. 학습률, 1차 모멘텀, 2차 모멘텀
- ``추가 조사 필요``

## 6.1.7 어느 갱신 방법을 이용할 것인가.
- 풀어야 할 문제가 무엇이냐에 따라 달라진다.
- 또한 하이퍼파라미터를 어떻게 설정하느냐에 따라서도 다른다.
- 많은 연구에서는 SGD를 사용한다.
- 각 상황에 맞게 잘 선택해서 사용 해야 한다.

# 6.2 가중치의 초기값
- 가중치의 초기값은 학습에 많은 영향을 미친다.

## 6.2.1 초기값을 0으로 하면?
- 초기값을 0으로 하면 학습이 올바로 이뤄지지 않는다.
- 정확히 가중치는 균일한 값으로 설정해서는 안된다.
- 그 이유는 바로 오차역전파법에서 모든 가중치의 값이 똑같이 갱신 되기 떄문이다.
- 이는 가중치들을 여러개 갖는 의미가 사라지게 되는 것이다.
- 가중치가 고르게 되는 상황을 막기위해 초기값을 무작위로 설정 해야 한다.

#### Weight Decay
- 오버피팅을 억제하는데 사용한다.
- 가중치 매개변수의 값이 작아지도록 학습하는 방법이다.
- 가중치 값이 작아지도록 하여 오버피팅이 일어나지 않도록 한다.


## 6.2.2 은닉층의 활성화값 분포
- 은닉층의 활성화값(활성화 함수 출력)의 분포를 관찰하면 정보를 얻을 수 있다.
- 각 층의 활성화값은 적당히 고루 분포되어야 한다.
- 층과 층 사이에 적당하게 다양한 데어티가 흐르게 해야 신경망 학습이 효율적으로 이뤄진다.

#### 가중치 표준편차가 1인 정규분포로 초기화할떄 활성화값 분포
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F34a15be0-4184-11ea-b59a-8d118c209878%2Fimage.png)

- 각 층의 활성화 값들이 0과 1에 치우쳐져 있다.
- 시그모이드 함수는 출력이 0과 1에 가까워 지면 그 미분은 0에 가까워 진다.
- 즉, 0과 1에 가까워 지면 기울기값이 0에 근사한 값이고 매개변수의 업데이트가 일어나지 않는다. 따라서 0과 1로 매개변수들이 분포하게 된다, 이러한 문제가 기울기 손실이다.
- 그래서 데이터가 0과 1에 치우쳐 분포 하게 되며 역전파의 기울기 값이 점점 작아지다 사라 진다. 이것을 기울기 손실 이라 한다.
- 한번 0과 1에 가까운 값이 됬다면 다음 오차역전파에서 변화가 없기 때문에 계속 0과 1에 가까운 값에 머무르게 된다.

#### 가중치 표준편차가 0.01인 정규분포로 초기화할떄 활성화값 분포
![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F4f890610-4184-11ea-b59a-8d118c209878%2Fimage.png)

- 0.5 부근에 값이 집중 되어 있다. 
- 0과 1로 치우치진 않았으니 기울기 손실 문제는 일어나지 않는다.
- 활성화 값들이 치우쳐다는 것은 다수의 뉴런이 거의 같은 값을 출력 하기 때문에 뉴런을 여러개둔 의미가 없어지게 된다.
- 활성화 값들이 추이치면 **표현력을 제한** 한다는 관점에서 문제가 생긴다.

#### Xavier 초기값
- Xavier 초기값은 일반적으로 딥러닝 프레임워크들이 표준적으로 이용하고 있다.
- 앞 계층의 노드가 n개 라면 표준편차가 1/sqrt(n) 인 분포를 이용하는 방법
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjc0/MDAxNTAxMTI2ODI0ODM3.LTKJgqWpZsgRgHX2-IhP5EM6MUo8tOTk_Fzl89bVV48g.ER6261bifFkOBBgxEwAamC3eDWBTABYd1r4gpXxrICsg.PNG.cjswo9207/fig_6-12.png?type=w2)
<br>
- 앞 층에 노드가 많을수록 대상 노드의 초기값으로 설정하는 가중치가 좁게 퍼진다.

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F175a4870-4185-11ea-9189-773222ee3eae%2Fimage.png)
- 층이 깊어 질 수록 형태가 일그러 지지만 앞에서 본 방식보다는 확실히 넓게 분포 되는것을 알 수 있다.
- 시그모이드의 표현력도 제한받지 않고 학습이 효율적으로 이루어질 것으로 예상 된다.
- tanh (원점 대칭 함수)를 사용하면 말끔한 종모양 분포가 된다.

## 6.2.3 ReLU를 사용할때의 가중치 초기값
- Xavier 초기 값은 선형활성화 함수(Sigmoid, tanh)에 적합한 방식이다.
- ReLU를 사용할때는 ReLU에 특화된 초기값을 사용 해야 한다.
- He초기값 이라 한다. 앞 계층의 노드가 n개 일때 sqrt(2/n)인 정규 분포를 사용 한다.
- ReLU의 음의 영역은 0 이라서 더 넓게 분포시기키 위해 2배의 계추가 필요하다고 해석할 수 있다???

![image](https://velog.velcdn.com/post-images%2Fdscwinterstudy%2F26142920-4186-11ea-8860-fb0fbd9ad9ee%2Fimage.png)

#### std = 0.0
- 각 층의 활성화 값들은 아주 작은 값이다.
- 신경망에 아주 작은 값이 흐른다는 것은 역전파 때 가중치의 값이 작아 진다는 것이며 학습이 거이 이뤄지지 않는다.

#### Xavier 
- 층이 깊어지면서 0 에 가까운 값으로 값이 치우친다.
- 이는 학습 할때 **기울기 손실** 문제를 일으킨다.

#### He 초기값
- 모든 층에서 균일하게 분포 되어 있다.


## 6.3 배치 정규화
- 각 층의 활성화를 적당히 퍼트리도록 **강제** 하도록 하는 아이디어가 **배치 정규화** 이다. 

#### 장점
- 학습을 빨리 진행할 수 있다.
- 초기값에 크게 의존하지 않는다. (가중치 조기값 선택 자유)
- 오버비팅을 억제한다. (드롭아웃 등의 필요성 감소)


## 6.3.1 배치 정규화 알고리즘
- 기본 아이디어는 각 층에서 활성화 값이 적당히 분포 되도록 조정하는 것이다. (Sigmoid 함수 사용시 한쪽으로 몰리는것을 방지)
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfOTUg/MDAxNTAxMTI4NTM3NTU2.aLAMgxDvmG9clAXUJ47gnN4DWmxvSr93uk5p_n6gDyMg.DqUt_OKyr-zNjpk_I7q6421W5aL7-NEhj8aTCMkWvXgg.PNG.cjswo9207/fig_6-16.png?type=w2)
- 배치 정규화 계층을 Affine 과 활성화 계층 사이에 삽입 한다.
- 학습시 미니배치 단위로 정규화 한다.
- 데이터 분포가 0 분산이 1이 되도록 정규화 한다.
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfNTgg/MDAxNTAxMTI4NjQyMDU1.eNQPdSS7ztsTBWW7LuPvQARlXE5fjHF3L7lcqZz-Pu4g.05WT2tsms0-Kl959MTDRiOze_0vPzK1ENmVW-WUT7M4g.PNG.cjswo9207/e_6.7.png?type=w2)

- B = [x1, x2, x3 ... xm]이라는 m개의 입력 데이터의 집합에 대해 평균(ub)와 분산(ab)를 구한다.
- 입력 데이터를 평균이 0 분산이 1이되게(적절한 분포가 되게) 정규화 한다.
- 단순히 미니배치 입력데이터를 평균 0 분산1인 데이터로 변환해서 활성화 함수의 앞 또는 뒤에 삽입 함으로써 데이터 분포가 치우치지 않게 한다.

#### 장점
- 학습을 빨리 진행할 수 있다, 학습률을 크게 해도 최적화가 가능하다.
- 초기값에 크게 의존하지 않는다.
- 오버피팅을 억제한다.(드롭아웃 등의 필요성 감소), 오버피팅은 가중치 매개변수의 값이 커서 발생하는 경우가 많다. 이러한 경우를 제거할 수 있다.


## 6.4 바른 학습을 위해
- 기계학습에서는 오버피팅이 문제가 되는 일이 많다.
- 오버피팅이란 신경망이 훈련 데이터에만 지나치게 적응되어 그외의 데이터에는 제대로 대응하지 못하는 상태를 말한다.
- 훈련 데이터에 포함되지 않는, 아직 보지 못한 데이터가 주어져도 바르게 식별해내는 모델이 바람직하다.

## 6.4.1 오버피팅

- 오버피팅은 주로 2가지 경우에 일어 난다.
    1) 매개변수가 많고 표현력이 높은 모델
    2) 훈련데이터가 적은 경우
![image](https://mblogthumb-phinf.pstatic.net/MjAxNzA3MjdfMjQ0/MDAxNTAxMTMzMDAzOTU1.5QbsUHjX2n5b8QGbP0TJRpwLbXou1YTIxGr8v9GwQggg.YX6aE7u6eIXfRRP8j8G499_r7qr_OFY_ipgEfy2MCl0g.PNG.cjswo9207/fig_6-21.png?type=w2)

- 훈련데이터는 거의 100%이나 시험 데이터는 큰 차이를 보인다.
- 이처럼 정확도가 크게 벌어지는 것은 훈련 데이터에만 적응 해버린 결과 이다.

## 6.4.2 가중치 감소(Weight Decay)
- 오버피팅의 억제용으로 Weight Decay라는 것이 있다.
- 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 패널티를 부과해 오버피팅을 억제하는 방법이다.
- 원래 오버피팅은 가중치 매개변수의 값이 커서 발생하는 경우가 많다.
- 신경망 학습의 목적은 손실함수의 값을 줄이는 것이다.(예측결과 정답이 되도록)
- 예를 들어 가중치의 제곱을 손실함수에 더한다.(일부러 손실을 크게 만든다.)
- 그러면 가중치가 더 커지는것을 억제 할 수 있다. (내 가중치가 더 커지면 오차가 커지기 떄문에 가중치가 커지는걸 방지할 수 있다.)
- 이 매개변수가 커질 수록 가중치가 커진다. (일부러 더함) 그렇기 때문에 이 매개변수는 더이상 커지면 안된다. (억제 된다.)

## 6.4.3 드롭아웃
- 신경망 모델이 복잡해지면 가중치 감소만으로 대응하기 어렵다.
- 드롭아웃 이란 기법을 이용한다.
- 뉴런을 임의로 삭제하면서 학습하는 방법이다.
- 훈련 때 은늑층의 뉴런을 무작위로 골라 삭제 한다.
- 삭제된 뉴런은 신호를 전달 하지 않는다.
- 추론 때는 모든 뉴런에 신호를 전달한다. 단, 각 뉴런의 출력에 훈려때 삭제 안한 비율을 곱하여 출력한다. ( 삭제 안된 만큼 학습이 더잘 됬으므로 더 많이 반영 되나 봄)
- 훈련 데이터에 대한 정확도가 줄어 드나, 훈련 데이터와 시험데이터의 차이가 줄어 든다. 

```앙상블 학습
- 개별적으로 학습시킨 여러 모델의 출력을 평균내어 추론하는 방식이다.
- 앙상블 학습은 드롭아웃과 밀접하다. 드롭아웃이 학습 때 뉴런을 무작위로 삭제하는 행위를 매번 다른 모델을 학습시키는 것으로 해석할 수 있다.
- 그리고 추론 때는 뉴런의 출력에 삭제한 비율을 곱합으로써 앙상블 학습에서 여러 모델의 평균을 내는것과 같은 효과를 얻을 수 있다.
```

# 6.5 적절한 하이퍼파라미터 값 찾기
- 각층의 뉴런수, 배치 크기, 학습률, 가중치 감소 등 다양한 하이퍼파라미터가 존재한다.

## 6.5.1 검증 데이터 
- **하이퍼파라미터의 성능을 평가 할 때는 시험데이터를 사용해서는 안된다.**
- 시험 데이터를 사용하여 하이퍼파라미터를 조정하면 하이퍼파라미터 값이 시험데이터에 오버피팅 되기 때문이다, 하이퍼파라미터의 값이 시험데이터에만 적합 하도록 조정되어 버린다.
- 다른 데이터에는 적응하지 못하니 범용 성능이 떨어지는 모델이 될 수 있다.
- 하이퍼파라미터 전용 확인 데이터를 Validation Data라고 부른다.
```
- Train Data : 매개변수 학습
- Validation Data : 하이퍼파라미터 성능 평가
- Test Dat : 신경망의 범용성 성능 평가
```
- 훈련데이터 중 20% 정도를 검증 데이터로 분리함으로써 검증 데이터를 확보할 수 있다.


## 6.5.2 하이퍼파라미터 최적화
- 하이퍼파라미터를 최적화할 때의 핵심은 하이퍼파라미터의 "최적 값"이 존재하는 범위를 조금씩 줄여간다는 것이다.
- 하이퍼파라미터를 골라낸 후, 그값으로 정확도를 평가하고 정확도를 잘 살피면서 이작업을 여러 번 반복하여 하이퍼파라미터의 "최적 값"의 범위를 좁힌다.
- 하이퍼파라미터 설정후 흑습 중 나쁠 듯한 값은 일찍 포기하는게 좋다.
- 에폭을 작게 하여, 1회 평가에 걸리는 시간을 단축한다.

1) 하이퍼파라미터 값의 범위를 설정한다.
2) 설정된 범위에서 하이퍼파라미터의 값을 무작위로 추출 한다.
3) 2단계에서 샘플링한 하이퍼파라미터 값을 사용하여 학습하고, 검증 데이터로 정확도를 평가 한다. (에폭은 작게)
4) 1단계와 2단계를 특정횟수 반복하여 그 정확도 결과를 보고 하이퍼 파라미터 범위를 좁힌다.

- **베이즈 최적화** 라는 더 엄밀하고 효율적으로 최적활르 수행하는 방법이 있다.