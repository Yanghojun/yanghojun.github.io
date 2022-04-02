---
layout: single
title:  "Pruning and quantization for deep neural network acceleration"
categories: [Paper-Review] # 홈페이지에서 카테고리를 통해 coding으로 지정되어있는 포스트들을 한번에 볼 수 있다
tag: [pruning, neural network, survey, trend]
permalink: /Pruning and quantization for deep neural network acceleration/
toc: true
toc_sticky: true
author_profile: false # 왼쪽에 조그마한 글 나오는지 여부
sidebar:
    nav: "docs"     # Navigation에 있는 docs
# search: false # 만약 이 포스트가 검색이 안되길 원한다면 주석해제
---
# Abstract

- 최신 기술, 기술들의 장단점, 현재의 압축 모델의 정확도를 여러 프레임워크에서 보고, 모델 압축을 위한 가이드를 제공할 것임

# Introduction

![](/images/2022-01-25-03-32-50.png)

- Novel Components
    - separable convolution, inception blocks, residual blocks 같은 효율적 블록 디자인 하는 것
    - layer 연결 방식 연구도 포함
- Network Architecture Search
    - 프로그래밍적으로 효율적인 네트워크 구조를 정의된 탐색영역(search space)에서 찾는 것
- Knowledge Distillation
    - Knowledge transfer에서 유래됐으며 larger model 처럼 역할하는 simple model 만드는 것