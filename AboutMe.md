---
layout: article
title: Jarvis Makers - 언젠가 인공지능 Jarvis를 상용화할 양호준입니다.
# permalink: /AboutMe/
permalink: /AboutMe/
article_header:
  type: overlay
  theme: dark
  background_color: '#203028'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(34, 139, 87 , .4), rgba(139, 34, 139, .4))'
    src: /images/back.jpg
aside:
    toc: true
---

# 학력 및 경력

1. <b style='color:blue'>인하대학교</b> 사범대학 부속 중학교 졸업  
2. <b style='color:blue'>인하대학교</b> 사범대학 부속 고등학교 졸업
3. <b style='color:blue'>인하대학교</b> 컴퓨터공학과 졸업
4. <b style='color:blue'>인하대학교</b> 전기컴퓨터공학과 인공지능 전공 재학중 (2021년 3월 ~ 2023년 2월 졸업 예정)
5. <b style='color:blue'>현대오토에버</b> 인공지능기술팀 (2023년 1월 ~ )

# GitHub 주소 :palm_tree: 
<div align='center' markdown='1'>
:palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree:  
:palm_tree::palm_tree::palm_tree: [https://github.com/Yanghojun](https://github.com/Yanghojun):palm_tree::palm_tree::palm_tree:  
:palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree::palm_tree:
</div>

# 프로젝트

<div class="grid-container">
<div class="grid grid--py-3">
<div class="cell cell--6">
<div>
<div>
<a href="#시각장애인용-임베디드-보조-시스템">
<div class="card card--clickable" style='margin: 1rem;'>
<div class="card__image">
<img class="image" src="/images/agriculture.gif"/>
<div align="center" markdown="1" style='color:mediumpurple'> 시각장애인용 임베디드 보조 시스템
</div>
</div>
</div>
</a>
</div>
</div>
</div>
<div class="cell cell--6">
<div>
<div>
<a href="#타이어-수명-예측-안드로이드-어플리케이션">
<div class="card card--clickable" style='margin: 1rem;'>
<div class="card__image">
<img class="image" src="/images/220902_tire_demo.gif"/>
<div align="center" markdown="1" style='color:mediumpurple'>타이어 수명 예측 안드로이드 어플리케이션
</div>
</div>
</div>
</a>
</div>
</div>
</div>
</div>
</div> 

<!-- 
<div class="cell cell--6">
<div>6 cells
</div>
</div>
<div class="cell cell--6">
<div>6 cells
</div>
</div>
-->

## 시각장애인용 임베디드 보조 시스템 
<div markdown='1' align='center'>기술스택<br>`JetPack 4.6.1`{:.error} `L4T(Linux for tegra`{:.info} `Docker`{:.info} `Python`{:.success} `TensorRT`{:.success} `Pytorch`{:.warning} `Google Speech To Text`{:.warning}
</div>
<hr>

**수행기간:** 2021.03.01 ~ 2022.10.30  
**참여역할:** 단독 수행 / 딥러닝 기반 Object 및 Text Detection&Recognition 기능 구현, 3D 구조물 설계 및 출력  
**내용:** Xavier 보드, Intel Depth 카메라, 마이크, 스피커, 휴대용 배터리를 활용해 시각장애인을 위한 Object 및 Text Detection&Recognition 기능을 수행하는 임베디드 시스템을 개발했습니다. 구현한 시스템으로 제 19회 임베디드소프트웨어 경진대회에 참가해 2등상을 단독 수상했습니다. 또한, 본 성과를 국내 등재학술지에 1저자로 논문을 게재하였습니다.
{:.success}


<div class="card">
  <div class="card__image">
    <p align="center"> <img src="/images/20221130144127.png" width="80%"> </p> 
  </div>
  <hr>
  <div class="card__content" align="center" markdown='1'>
  전반적인 프로젝트 결과물
  </div>
</div>

<div class="card">
  <div class="card__image">
    <p align="center"> <img src="/images/20221130135528.png" width="80%"> </p> 
  </div>
  <hr>
  <div class="card__content" align="center" markdown='1'>
  직접 학습 데이터 3000장을 제작하여 학습에 사용했습니다. 그러나, 실내에서 촬영한 데이터가 대부분이여서 실증을 대비하기 위한 야외 테스트 데이터셋에서의 성능이 너무 낮았습니다. (Recall: 0.17, Precision: 0.47)  
  이를 보완하기 위해, 야외에서 데이터를 1000장 추가로 제작하고, 데이터 어그멘테이션을 진행해서 총 20,000장의 학습 데이터로 성능을 끌어올렸습니다. (Recall: 0.56, Precision: 0.79)
  </div>
</div>

<div class="card">
  <div class="card__image">
    <p align="center"> <img src="/images/20221130140756.png" width="100%"> </p>
  </div>
  <hr>
  <div class="card__content" align="center" markdown='1'>
  전체 프로세스 과정  
  1.착용자가 찾고자 하는 물건을 마이크에 입력시킵니다.  
  2.현재 시선 방향 기준으로 위치 정보, 거리 정보를 스피커로 출력합니다.  
  3.텍스트 정보가 들어왔으며, 사전에 등록된 텍스트일 경우 시각장애인을 위한 관련 안내를 스피커로 출력합니다.
  </div>
</div>

&nbsp;
&nbsp;

## 타이어 수명 예측 안드로이드 어플리케이션
<div align='center' markdown='1'>
기술스택<br>`Kotlin`{:.info} `Java`{:.info} `Android`{:.success} `Python`{:.success} `Pytorch`{:.warning} `Pruning`{:.warning}
</div>
<hr>

**수행기간:** 2021.03.01 ~ 2022.10.30  
**참여역할:** 안드로이드 어플리케이션 개발, 세그멘테이션 모델 학습 및 평가  
**내용:** 딥러닝 모델을 활용한 Segmentation, Regression 기능으로 타이어의 트레드 깊이를 예측하는 안드로이드 어플리케이션을 개발했습니다. 임베디드 시스템 환경임을 고려하여 모델 경량화를 진행했습니다. 이를 통해 비슷한 성능 대비 35% 추론 속도 향상, 60% 메모리 절약 이점을 확보했습니다.
{:.success}

<div class="card">
  <div class="card__image">
    <p align="center"> <img src="/images/20220921220601.png" width="80%"> </p>
  </div>
  <hr>
  <div class="card__content" align="center" markdown='1'>
  개발한 어플리케이션으로 타이어 데이터셋을 구축하면서, 불편했던 UI 부분을 개선시켰습니다.
  </div>
</div>

<div class="card">
  <div class="card__image">
    <p align="center"> <img src="/images/20220921215448.png" width="100%"> </p>
  </div>
  <hr>
  <div class="card__content" align="center" markdown='1'>
  전체 어플리케이션 흐름도
  </div>
</div>

<div class="card">
  <div class="card__image">
    <p align="center"> <img src="/images/20220921215756.png" width="100%"> </p>
  </div>
  <hr>
  <div class="card__content" align="center" markdown='1'>
  학습 완료된 Regression 관련 딥러닝 모델에 L2 Norm 기반의 채널 필터링 작업을 진행해 경량화를 수행했습니다.
  </div>
</div>

&nbsp;
&nbsp;

## 딥러닝 기반 기상 자료 이상 탐지 알고리즘 구현
<div align='center' markdown='1'>
기술스택<br>`Python`{:.success} `Pytorch`{:.warning} `Scikit-learn`{:.error}
</div>
<hr>

**수행기간:** 2021.03.01 ~ 2022.02.20  
**참여역할:** SOTA 딥러닝 모델 적용, 모델 레이어 수정 및 학습 데이터 부족 문제 해결  
**내용:** beatGAN 모델을 활용해 기상 자료 이상 탐지 알고리즘을 구현했습니다. 레이어를 대칭적으로 확장시켜 한번에 학습 가능한 데이터 길이를 늘리고, 모델의 생성적 특성을 활용해 학습에 쓰이는 데이터 수를 늘렸습니다. 이를 통해 초기 모델의 F1-SCORE 0.66에서 0.84로 향상시켰습니다. 본 성과를 국내 등재학술지에 1저자로 논문을 게재하였습니다.
{:.success}

&nbsp;
&nbsp;

## 딥러닝 기반 인서트 결함 검사
<div align='center' markdown='1'>
기술스택<br>`Python`{:.success} `Pytorch`{:.warning}
</div>
<hr>

**수행기간:** 2022.03.01 ~ 2022.06.24  
**참여역할:** 딥러닝 모델을 활용한 이상 탐지 알고리즘 구현  
**내용:** 영상 Cropping과 여러 딥러닝 SOTA 모델을 활용해 비지도 학습 방식으로 인서트의 작은 결함을 검출해냈습니다. BFS 알고리즘을 통해 비정상 영역 필터링 작업을 거쳐 초기 F1-Score 0.64를 0.84로 향상시켰습니다.
{:.success}

&nbsp;
&nbsp;

## 강화학습 및 ROS를 활용한 길 찾기 알고리즘 성능 비교
<div align='center' markdown='1'>
기술스택<br>`ROS`{:.warning} `Python`{:.success} 
</div>
<hr>

**수행기간:** 2021.10.25 ~ 2021.11.30  
**참여역할:** DQN, Q-learning 학습 코드 적용, 성능 비교 실험 진행  
**내용:** 가상 시뮬레이터에서 터틀봇 객체를 학습시킨 후 실제 터틀봇에 학습한 가중치 파일을 로드하여 두 강화학습 알고리즘(DQN, Q-learning)에 대한 길 찾기 성능을 비교했습니다.
{:.success}

# 주요 논문 성과

<div class="item">
<div class="item__image">
<img class="image image--xl" src="/images/paper1.png"/>
</div>
<div class="item__content">
<div class="item__header">
<h4 align='center'>KCI 1저자 게재</h4>
<div class markdown='1'>
- 제목: 적대적 생성 신경망을 활용한 비지도 학습 기반의 대기 자료 이상 탐지 알고리즘 연구
- 내용
  - 시계열 이상탐지를 BeatGAN 모델을 이용해서 진행함
  - 입력 데이터 사이즈 늘리고, 그에 따른 model capacity를 키우기 위해 layer를 추가하여 성능 개선
  - GAN 모델의 데이터 생성 특성을 이용해 부족한 학습 데이터 문제 해결 및 성능 개선
</div>
</div>
<div class="item__description">
</div>
</div>
</div>

<hr>

<div class="item">
<div class="item__image">
<img class="image image--xl" src="/images/paper2.png"/>
</div>
<div class="item__content">
<div class="item__header">
<h4 align='center'>KCI 1저자 게재</h4>
<div class markdown='1'>
- 제목: 시각장애인을 위한 딥러닝 기반의 실시간 임베디드 보조 시스템 개발에 관한 연구
- 내용
  - Jetson 보드, 마이크, 스피커, Depth 카메라, DeepLearning model, OCR 기능을 모두 결합해 시각장애인 보조를 위한 임베디드 시스템을 만듦
  - TensorRT 적용을 통한 가속화 진행
</div>
</div>
<div class="item__description">
</div>
</div>
</div>

# 수상 실적
<div class="card">
<div class="card card--clickable">
<div class="card__image">
<img class="image" src="/images/price1.png" />
<div class="overlay overlay--top" markdown='1'>
<h2>제 19회 임베디드 소프트웨어 경진대회 종합 2등</h2>
</div>
<div class="card__content">
<h3>한국전자기술연구원 원장상 수상</h3>
<div markdown='1'>
- 날짜: 2021년 12월 21일  
- 공모전: 제 19회 임베디드 소프트웨어 경진대회 (산업통산자원부 주최)   
- 팀명: AI Helpler (나 혼자 산다)
- 내용: 딥러닝 모델은 물체 검출 및 추정을 위한 Yolov5 모델과, 글자 인식 및 추정을 위한 EasyOCR 모델을 사용하였습니다. 마이크, 스피커, Jetson Xavier, 휴대용 배터리, Intel D455 카메라, 3D 프린팅 안경을 통해 시각장애인을 보조하기 위한 장착형 임베디드 시스템을 구축하였습니다. 또한, TensorRT 적용을 통해 시스템의 실시간성을 확보하였습니다.  마이크를 통해 시각장애인분이 찾고자 하는 물건을 입력하면 현재 시선의 방향을 중심으로 해당 물건의 거리 및 방향 정보를 스피커를 통해 출력합니다. 또한, 글자 정보를 가까이 카메라 가까이에 입력시킬 경우, 시각장애인에게 도움이 될 법한 사전 정보를 음성 파일을 통해 출력합니다.  본 시스템을 통해 제 19회 임베디드 소프트웨어 경진대회에서 2등을 차지하여 단독으로 우수상을 수상하였습니다.
<div align="center" markdown="1">
<h3>대회 제출 영상</h3>
</div>
<div align='center'>
<iframe width="560" height="315" src="https://www.youtube.com/embed/cj4JxTMyRA4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
</div>
</div>
</div>
</div>
</div>
&nbsp;
&nbsp;
&nbsp;
<div class="card">
<div class="card card--clickable">
<div class="card__image">
<img class="image" src="/images/20220929215822.png" />
<div class="overlay overlay--top" markdown='1'>
<h2>한국ITS학회 우수논문상</h2>
</div>
<div class="card__content">
<h3>수상 내용</h3>
<div markdown=1>
- 날짜: 2022년 6월 17일
- 내용: Car Detection 및 Super Resolution을 통한 Multi Object Tracking 성능 향상 방안 설계 및 실험결과 제시를 통해 우수논문상을 수상하였습니다.
</div>
</div>
</div>
</div>
</div>
&nbsp;
&nbsp;
&nbsp;
<div class="card">
<div class="card card--clickable">
<div class="card__image">
<img class="image" src="/images/20220929214012.png" />
<div class="overlay overlay--top" markdown='1'>
<h2> 한국인공지능융합기술학회 우수논문상</h2>
</div>
<div class="card__content">
<h3>수상 내용</h3>
<div markdown=1>
- 날짜: 2021년 8월 26일  
- 내용: 대기오염측정망 자료확정 알고리즘 개발을 위한 비지도학습 적용 논문으로 우수논문상을 수상하였습니다.
</div>
</div>
</div>
</div>
</div>


