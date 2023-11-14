---
layout: article
title:  "Juneer_Deeplearning_Cookbook 만들기"
category: 객체지향 프로그래밍
tag: [deeplearning, 상속, 객체지향]
permalink: /JuneerDeeplearningCookbook/
show_author_profile: true
aside:
    toc: true
sidebar:
    nav: "study-nav"
---

> 본 포스터는 한 Repository를 객체지향적으로 설계 및 구현해보고자 작성하는 포스트 입니다.
> 본 Repository의 링크 입니다. [https://github.com/Yanghojun/Juneer_Pytorch_CookBook](https://github.com/Yanghojun/Juneer_Pytorch_CookBook)

<p align="center"> <img src="/images/화면 캡처 2023-03-29 180953.jpg" width="80%"> </p>
<p align="center" markdown="1">
하나의 main 함수에서 특정 상황에서는 pytorch 프레임워크만 쓰고, 특정 상황에서는 keras 프레임워크만 쓰게 하고 싶어서 위와 같은 그림을 그리며 고민했었다.
</p>

그러나 모든 `dataloader`, `model` 코드에 `if pytorch, else`이 붙게되서 비효율적이라는 느낌을 받았다. 상속과 같은 방법을 써서 어떻게 해결할 수 있지 않을까 싶었지만...

<p align="center"> <img src="/images/화면 캡처 2023-03-29 192906.jpg" width="80%"> </p>
<p align="center" markdown="1">
이렇게 보니 `if pytorch, else`문이 정말 끝도없이 나올 수 밖에 없다. 데이터 전처리나 모델 정의 부분이 달라지기 때문에 상속을 통해 method override를 하더라도 항상 재정의를 `if pytorch, else`를 포함해서 해야한다. 비효율적인 느낌이 난다.
</p>

그래서 그냥 Pytorch 프레임워크를 쓰는 Repository, Keras 프레임워크를 쓰는 Repository로 나누기로 했다.

