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

- depth_gt: PCD, CamIntrinsic(calibration 파일) 이용해서 2d projection한 <span style="color:red">Grount Truth</span>
- depth_img: 일부러 오차 정보를 행렬곱시킨 PCD, CamIntrinsic(calibration 파일) 이용해서 2d projection한 데이터

<p align="center"> <img src="/images/image-7.png" width="100%"> </p>
<div align="center" markdown="1">
depth_gt 이미지
</div>

<p align="center"> <img src="/images/image-8.png" width="100%"> </p>
<div align="center" markdown="1">
depth_img 이미지
</div>

이 둘의 이미지가 조금 달라야 할 것 같은데... 왜 똑같지..

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

#### Quaternion to RotationMatrix

```python
def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat
```

### Loss 함수

```python
class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot      # pose_loss: regression loss

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)      # RT_target과 RT_predicted를 행렬곱하는게 무슨 의미인거지?

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)       # 무조건 100. 이상의 값으로 둔다는 것 같음.
            point_clouds_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = (1 - self.weight_point_cloud) * pose_loss +\
                     self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]

        return self.loss
```

- 의문점들
  1. Ground Truth, Predicted result를 비교해서 Pose loss(Regression loss)를 구했음. 근데, 같은 정보(Ground Truth, Predicted result)를 활용해서 Point Cloud Distance loss를 구하는 수학적 의미가 있을까?
     - 수학을 못하는 응애의 시선으론 두 loss가 거의 동일한 역할을 하는것으로 보임. 상호 보완이 되는 뭔가가 있는것인가?
  2. 위 1과 연관된 의문으로 `RT_total` 이 유도되는 코드를 보면 `RT_target`과 `RT_predicted`를 행렬곱하고 있는데 이는 `RT_predicted`를 error로 간주해서 이 error만큼 다시 회전과 이동을 진행시켜준다고 받아들이면 될까?.. 오 맞는듯..??? $$ \rightarrow $$ 다시 생각해보니 아닌듯.. self.transl_loss쪽 보면 거리가 서로 가까워야 loss 떨어질게 명확히 보임. 즉 transl_err가 err를 의미하는건 아닐거로 생각됌.