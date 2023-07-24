---
title: LCCNet 논문 리뷰
author: cotes
date: 2019-08-11 00:34:00 +0800
categories: [연구개발자로의 길, 논문 리뷰(코드 레벨 분석)]
tags: [LCCNet, Calibration, 카메라-라이다-정합, Deep-Learning]
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

- Te