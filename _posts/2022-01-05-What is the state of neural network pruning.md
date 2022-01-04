---
layout: single
title:  "What is the state of neural network pruning?"
categories: Paper-Review # 홈페이지에서 카테고리를 통해 coding으로 지정되어있는 포스트들을 한번에 볼 수 있다
tag: [Pruning, Neural network, state-of-art]
permalink: /paper1/
toc: true
author_profile: false # 왼쪽에 조그마한 글 나오는지 여부
sidebar:
    nav: "docs"     # Navigation에 있는 docs
# search: false # 만약 이 포스트가 검색이 안되길 원한다면 주석해제
---
# Abstract
- Pruning에 대한 benchmark가 어려워서 지난 30년동안 어느정도의 성과가 있었는지 정량적 평가가 어려움
- ShrinkBench라는 오픈소스 활용해서 표준화된 방법으로 Pruning 방법들 비교하고자 함
# Introduction
- 머신러닝이 딥러닝에 힘입어 주목받고 있으며 핸드폰같이 제한된 환경에서도 신경망을 사용하는 경우가 생김
- parameter 줄이는 대표적인 방법인 pruning을 조사할 것임
- 81개의 논문을 조사했으며 대부분의 논문이 다른 프루닝 방법과의 비교나 하이퍼 파라미터 구체화나 한개 방법정도와만 비교한 정도라서 성능적 우수성을 객관화 한 논문이 거의 없었음
- ShrinkBench 이용해서 알아볼것임