# Probability(확률)
- 주어진 확률 분포가 있을 때, 관측값 혹은 관측 구간이 분포 안에서 얼마의 확률로 존재하는 가를 나타내느 값이다.
- 확률 분포(Probability Distribution)을 고정하고 그때의 관측 X 에 대한 확률을 구하는 것이다.
- 확률 = P(관측 값 X | 확률 분포 D)

<br>

![image](https://t1.daumcdn.net/cfile/tistory/99CB8E365B20D66C02)
- 평균 32, 표준 편차 2.5를 갖는 정규 분포라고 가정하고 32-34 사이로 관측될 확률은 빨간 영역과 같다.
- 즉 **어떤 고정된 분포**에서 이것이 관측될 확률 이다.

# Likehood(가능도)
- 어떤 값이 관측 되었을 때, 이것이 어떤 확률 분포에서 왔을지에 대한 확률 이다.
- 관측치와 얼마나 유사한 확률 분포 인지 를 나타낸다.
- 가능도 = L(확률분포 D | 관측 값 X)

## 예시
- 쥐의 무가게 34g이고 이 관측 결과가 정규분포(m=32 / sd=2.5)에서 나왔을 확률은 0.12이다.
- 관측 값이 고정되고, 그것이 주어졌을 때 해당 확률 분포에서 나왔을 확률을 구하는 것이다.
- 빨간 십자 마크(0.12)가 가능도가 된다.
<br>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F994A81365B20D66C20)

- 평균이 34이고 표준편차가 2.5일때 확률 분포에서 34g의 가능도가 더 높아진다.
<br>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99B30D365B20D66D04)

# Maximum Likelihood
- 각 관측값에 대한 총 가능도(모든 가능도의 곱)가 최대가 되게하는 분포를 찾는것이다.
<br>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99B607435B20DEC120)
- 여러개의 관측 값이 있을때, 이렇게 관측 될 가능성이 가장 큰 확률 분포는 무엇인지 찾는 것이 Maximum Likehoood이다.

![image](https://t1.daumcdn.net/cfile/tistory/99E1CD435B20DEC11D)
- 평균이 왼쪽으로 치우친 정규 폰포 일때 총 가능도를 구할 수 있다.
- 관측 치를 x라고 하고 확률 밀도 함수의 f(x) 값이 가능도 이다.
- 가능도의 총 합은 검은 점과 유사할 것이다.

![image](https://t1.daumcdn.net/cfile/tistory/99CDF1435B20DEC20A)
- 확률 분포의 평균을 변화 했을때 가능도의 총 합의 변화를 확인할 수 있다.
- 즉, 수집한 관측값들이 나올 수 있는 가장 높은 확률 분포는 가능도 총곱이 가장큰 확률 분포이다.
- 각 데이터 샘플에서 후보 분포에 대한 높이(likelihood 기여도)를 계산해서 다 곱한것이다.
- 덧샘을 하지 않고 곱해주는 것은 모든 데이터는 독립적으로 일어난 확률이기 때문이다.

<br>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fx9qXc%2FbtrJ9vhICq9%2Fdy7uFuOMNJYxEo4KeKYHBk%2Fimg.png)
- 위 식은 Likelihood Function 이로가 하고, 보통은 자연로그를 이용해 아래와 같이 log-likelihood function 을 이용한다.

<br>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHR2mD%2FbtrKei2wcz7%2F2dbPAA5Sd7X1AGdDEGt6x1%2Fimg.png)
- log 함수는 단조 증가 함수이기 때문에, likelehood function최대값과 log-likelehood function의 최대값은 동일 하다.
- 계산의 편의를 위해 log-likelihood 최대값을 찾는다.
- log의 특성으로 곱이 덧샘으로 변경되어 계산하기 더 수월하다.

## 최대값 찾는법
- likelihood 함수의 최대 값을 찾는 보편적인 방법은 미분계수를 이용하는 것이다.
<br>

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FrqDdZ%2FbtrKayrJ3wB%2FDg7rYWLDLyEx1PObbbTIKK%2Fimg.png)
- 찾고자 하는 파라미터 𝜃(확률분포)에 대해서 다음과 같이 편미분 하고 그 값이 0이 되도록 하는 𝜃를 찾는 과정을 통해 최대값을 찾을 수 있다.
- 이때 𝜃는 평균과 표준편차 2개의 변수 이다.


# 머신러닝의 어느 부분에서 사용?
- 모델의 성능을 평가하는 단계에서 사용한다.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdlbwJj%2Fbtru4iRHlRA%2FZAWZ1gzk3SYkq3IDtwk3vk%2Fimg.png)

- 1이 정답인 데이터의 모델의 마지막 Softmax layer의 확률 값을 표현 했을때, 가장 정답에 가까운 분포를 평가 해야 한다.
- 모델 C가 데이터를 가장 잘 설명하는 Distribution이며 Likehood가 가장 높은 분포이다.
- 따라서 이 Likehood를 최대화 시키는게 ML의 목적이다.
- A같은 모델을 C로 만들기 위해 Loss로 설정하고 진행한다.(Cross Entropy)

## Cross Entropy
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fp2C7P%2Fbtru8HXchVl%2F46ykRfQv9W35CqYqcSZICK%2Fimg.png)

- Cross Entropy공식을 이용하여 모델의 성능을 평가할 수 있다.

#### 참조
- [링크1](https://huidea.tistory.com/276)
- [링크2](https://jjangjjong.tistory.com/41)
- [링크3](https://studyingrabbit.tistory.com/66)
- [링크4](https://mac-user-guide.tistory.com/182)
- [링크5](https://angeloyeo.github.io/2020/07/17/MLE.html)