---
layout: article
title:  "Pytorch Cuda 설치"
category: [Pytorch]
tag: [CUDA, NVCC, CUDA-TOOLKIT]
show_author_profile: true
permalink: /PytorchCudaInstall/
aside:
    toc: true
sidebar:
    nav: "study-nav"
---

> 본 포스터는 많은 사람들이 오해하는 것으로 보이는(제가 틀린걸수도 있어요.. ㅋㅋㅋ) Pytorch 설치 부분을 바로잡고자 작성한다.

# GPU 사용이 가능하도록 Pytorch 설치하기

많은 블로그 글들이 Nvidia 공식 홈페이지에서 `Nvidia Driver`, `CUDA`, `cuDNN`을 직접 설치한 후 `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia` 와 같은 명령어로 Python 환경에 관련 라이브러리를 설치하게끔 한다.  

예전 Pytorch 프레임워크를 사용할 때는 그랬을지 모르겠지만 지금은 사용자가 **별도로 CUDA, cuDNN을 설치할 필요가 없다.**

[파이토치 공식 홈페이지](pytorch.org)에서 제시하는 `pip3`, `conda` 관련 명령어만 실행해주면 `CUDA Toolkit`[^1]을 사용자 환경에 맞게 설치하기 때문에 별도로 `CUDA`, `cuDNN`등을 설치할 필요가 없다.

다시 정리하면 GPU 사용이 가능하도록 Pytorch를 설치하기 위해선

1. `conda` or `pip3` 환경
2. 최신 `Nvidia Driver` 설치

만 진행해두면 된다.

믿기지 않는다면 아래 2023년 1월 24일자 글을 보자.

<p align="center"> <img src="/images/230418-0003.jpg" width="100%"> </p>
<div align="center" markdown="1">
그림 1
</div>

# Nvidia 공식 홈페이지에서 받은 CUDA Compiler와 Pytorch 연동하기

위 [그림 1](/images/230418-0003.jpg)에서 이미 본 사람도 있겠지만, 로컬 컴퓨터에 `CUDA 컴파일러`를 설치한 사람의 경우는 `build Pytorch from source` 작업을 진행해야 한다.

전문적인 AI 개발자가 아닌 이상, 대부분 유저는 [GPU 사용이 가능하도록 Pytorch 설치하기](#gpu-사용이-가능하도록-pytorch-설치하기) 부분만 참고하면 될 듯 하다.

`Local CUDA Compiler`를 연동하는 내용은 향후 작성해 보겠다.

[^1]: GPU 프로그래밍을 위한 도구 모음으로 다음과 같은 Tool 들이 있음. <br> `CUDA C/C++ compiler`, `CUDA runtime lib`, CUDA DEV-kit, CUDA math 라이브러리, `cuDNN`