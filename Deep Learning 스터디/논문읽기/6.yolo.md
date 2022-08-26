## Object Detection
- 물체의 위치 예측과 어떤 물체 인지 분류

## 2-stage Object Detector
- 객체 탐지까지 과정을 2 단계로 분리됨
1) 물체의 영역을 예측하는 부분
2) 물체의 Label을 예측하는 부분
- detection 성능은 높지만, Inference 속도가 높다.

## 1-stage Object Detector
- input Image -> Feature Extraction -> boundbox, label 예측
- 2-stage는 물체의 Bounding Box 예측 후보들을 얻어 낼 수 있지만 1-stage는 그런 과정이 없다.
- 셀 (feature map의 1pixel) 별로 3x3 이나 5x5만큼 bound box를 미리 정해서 예측 한다. 이것을 anchor box라 한다.

## Yolo
- Yolo는 정확한 성능 보다는 Real-Time 을 위한 속도에 초점
- 실제 속도에 적용하기 위해서는 빠른 속도가 중요하다.
- Single GPU에서 사용가능

## Yolo v3
- Bounding Box를 예측 하기 위해 사전에 Anchor Box를 미리 설정해 예측 한다.
- 3개의 feature map에 3개의 Anchor box를 구한다.
- Anchor box 크기 : 데이터 셋을 K-mean clustering 하고 데이터 셋의 평균점에 있는 Object의 크기를 계산하여 사용한다.( 총 9개 로 클러스터링)
- 학습을 통해 bound box를 예측해 가며 크기를 변경한다.
- 여러개중에 하나를 찾는 Softmax가 아닌 각 클래스가 맞는지 아닌지 맞는지로 Binary Classification으로 적용 했다.
- Darknet-53인 백본을 사용했다, residual connection, bottle neck 기법을 추가 적용했다.
- 3개 scale map에 대해서 Object detection을 수행한다.

## Yolo v4
- Back bone : classification Task 로 학습된 Feature Map을 뽑는 모델
- Neck : Back bone과 예측으 연결 하는 부분 (Yolo v3에서 여러 Scale Map에서 Dense Layer로 연결하는 부분)
- 224, 256 => 512 크기로 학습
- layer수를 늘렸으며, Paramter 수를 키웠다. 속도에서 손해를 보지 않기위해 CSPNet기반의 CSPDarkNet53을 사용하였다.