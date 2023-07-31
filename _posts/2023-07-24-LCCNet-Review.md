---
title: LCCNet 논문 리뷰
date: 2023-07-24
categories: [연구개발자로의 길, 논문 리뷰(코드 레벨 분석)]
tags: [LCCNet, Calibration, 카메라-라이다-정합, Deep-Learning]
math: true
---

# LCCNet


## 코드 분석
- 23.07.24 이슈 정리
  - `save_for_backward` 에러 발생
    - pytorch 프레임워크 찾아보니 `save_for_backward` 함수는 submodule들의 weight 초기화를 위한 **함수**를 입력으로 주면 되는걸로 보임. 근데 코드상으론 **Tensor**를 주고있는데 에러가 안나서 <u>의아하긴함..</u>
  - Pytorch Official Github는 Cuda 11 버전을 지원하지 않아 코드를 실행하기 막막했으나, `callzhang`이라는 사람이 Cuda 11 이상 버전을 위한 branch를 생성해놔서 이를 기반으로 코드를 분석중이다 -> [Repo](https://github.com/callzhang/LCCNet/tree/main)
    ```python
    corr6 = self.corr(c16, c26)     # 처음엔 값이 잘 들어가다가 4번째로 값이 들어갈 때 c26이 비어있다며 에러남.
                                    # 그러나, 디버깅을 해봐도 계속 shape은 정상적으로 출력됌. 환장하겠음.
                                    # pytorch framweork 코드 분석으로 save_backward 에러는 해결한걸로 보임
    ```

- 23.07.30 해결 사항
  - 위 언급한 `save_for_backward` 에러에 대해 아래와 같이 해결함

    ```python
    # ctx.save_for_backward(self, input1, input2)   # 기존 코드. save_for_backward 함수는 Tensor만 인자로 넘겨야 하는데 Python object를 넘기고 있어서 계속 에러 났었음.
    ctx.self = self                                 # 이 방법으로 아주 쉽게 backward 함수에서 접근 가능하도록 만들 수 있었음
    ctx.save_for_backward(input1, input2)           
    ```
  - 내 Repository에 올려둠 -> [Repo](https://github.com/Yanghojun/LCCNet.git)
  - 현재 PCD들이 생성되고 있지만 Node 14 버젼이어야 pcd viewer를 설치할 수 있다. 근데 이 블로그 설치하려고 node 18 버젼 설치했던거라 node 다운그레이드는 나중에 하자.


### 데이터 전처리

#### Depth LiDAR Image 생성

- 이미지 시각화

```python
plt.imsave('./pic.png', depth_gt.cpu().permute(1,2,0).squeeze(), cmap='gray')   # depth 이미지 한장 생성해보기 위해 작성한 코드
# 1. gpu에서 cpu로 부름
# 2. height, width, channel 순으로 배치
# 3. gray 스케일로 이미지 뽑을것이므로 squeeze()를 통해 한 차원 축소
```

<p align="center"> <img src="/images/image-6.png" width="80%"> </p>
<div align="center" markdown="1">
결과 이미지 (Depth LiDAR Image)
</div>

- Projection 코드 분석
  - 필요 이론: Perspective Projection Transformation [다크프로그래머](https://darkpgmr.tistory.com/82)
    - 변환 매트릭스가 왜 $$ \left[ \begin{matrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1/d & 0 \end{matrix} \right] $$ 인지 아직 명확하지 않음.  
      <span style="color:blue">Sol(23.07.31):</span> 동차좌표계이므로 $$ \left[ \begin{matrix} {dx_c}\over{z_c} \\ {dy_c}\over{z_c} \\ 1 \end{matrix} \right] = \left[ \begin{matrix} x_c \\ y_c \\ dz_c \end{matrix} \right]$$ 이다. 그리고 동차좌표계는 한 좌표를 무수히 많은 다른 좌표로 표현이 가능한데, 이를 **스케일링 팩터**를 사용해서 가능하게 한다. 


```python

```
