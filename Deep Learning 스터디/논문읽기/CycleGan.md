## GAN의 목표와 Image-to-Image Translation 소개
- GAN을 학습하여 이미지 분포를 근사하는 모델 G를 학습할 수 있다.
  - 모델 G가 잘 작동한다는 의미는 원래 이미지들의 분포를 잘 모델링 할 수 있다는 것을 의미한다.

## Generative Adversarial Network
- GAN은 생성자(Generator)와 판별자(Discriminator) 두개의 네트워크로 구성된다.
  - Discriminator는 특정 이미지의 진짜/가짜 여부를 판별 한다.
  - Generator는 판별자를 속일 수 있는 이미지를 생성 한다.
  <br>
  ![image](https://www.researchgate.net/publication/340481789/figure/fig1/AS:877673731596293@1586265132992/The-original-generative-adversarial-network-GAN-model-In-simple-terms-G-wants-to.png)
- Generator와 Discriminator 가 서로 경쟁적으로 학습 하며 결과적으로 Generator는 실제와 같은 이미지를 생성한다.

## Pix2Pix
- 한장의 이미지가 주어 졌을떄 그 이미지의 특성을 다른 특성을 갖게 하는 Task에 사용될 수 있다.
- 입력이미지 자체가 Condition인 Conditiional GAN이라 할 수 있다.
- 사진이미지-그림이미지 처럼 한쌍으로 들어간다, 미리 데이터 셋을 알고 있고 pair이미지에 대해서 변환이 가능하다.
- 서로다른 두 도메인 X,Y의 데이터 두개를 한쌍으로 묶어 학습을 진행한다, 이러한 데이터 셋을 구성하기 어렵다.
- ![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdPe3rn%2FbtqTF4bgPvX%2FhlKz7RF0u3mqSt8PpK2ka0%2Fimg.png)
- Blurring 으로 표현된 Output이 나온다, Generator는 Input이미지만 면 깃털 색이 어떤 색인지, 벽돌이 어떤색일지 알 수 없기 때문에 Loss가 너무 커지지 않도록 애매한 중간값을 택하는 경향 때문에 애매한 결과물을 도출 하는 것이다.

## Conditional GAN
- 데이터의 모드를 제어할 수 있는 조건(Label 같은) 정보를 함꼐 입력 하는 모델
- cGAN 은 ground truth를 추가로 Generator에 제시해 Input image를 Ground Truth와 연관된 이미지로 전환 할 수 있다.

## CycleGan
- Unpaired 데이터셋 상황에서 Cylce 손실을 이요하여 학습 하는 GAN 네트워크
- 매칭되는 Y 없이 단순히 입력 이미지 x의 특성을 타겟 도메인 Y의 특성으로 바꾸고지 한다.
<br>
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkZrQ1%2Fbtq2GW83ez7%2FdGlDYgz5Qcjo7gzramdtX0%2Fimg.png)
- 이떄, GAN Loss만 사용하게 되면 G는 X가 어떤것이 들어와도 X Content 정보를 아에 바꾼 Y 도메인중 하나의 이미지를 제시할 수 도 있다.
- input의 특징을 모두 잊어 버리고 똑같은 출력을 생성하는 문제를 Mode Collapse라 한다.

<br>
![image](https://wikidocs.net/images/page/146366/1234.png)
(b)의 경우를 forward consistency라고 하고 (c)와 같은 경우를 backword consistency라 고 한다. Input Image가 G,F를 한바퀴 돌아 생성된 입력이미지와 Output이미지의 차이를 Cycle Consistency Loss라고 한다.
- 그러므로 X Content를 유지하는 Loss가 추가적으로 필요하다.
- Cycle-consistency loss : 생성한 이미지 G(x)가 다시 원본 x로 재구성 할 수 있도록 한다.
- 2개의 변환기가 존재 한다. 입력이 2개의 변환기를 순환 했을때 다시 원본 x와 비슷한 이미지가 나올 수 있도록 학습한다.

## 한계
- 데이터 셋이 불안전 하면 이미지르 제대로 생성할 수 없다.
  - 사람을 태운 사진이 없다면 사람까지 얼룩말로 바꿔버린다.
- 이미지의 모양까지는 바꾸지 못한다.
  - 사과를 오렌지로 바꾸어도 색만 바뀌지 모양을 바꾸지는 못한다.