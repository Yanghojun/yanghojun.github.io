---
layout: article
title:  "Pull request"
category: [깃허브]
tag: [Git, 협업, Pull request]
permalink: /Pull request/
aside:
    toc: true
sidebar:
    nav: "study-nav"
---

# 그게 머얌?
- Pull request = Merge request로 생각하면 편하다
  - 나의 branch를 remote의 master branch에 merge하고 싶을 때 담당자에게 `검토` 받는것
  ![](/images/2022-02-18-00-41-33.png)
  - 상황에 따라 2가지 Pull Request로 나뉨
    1. 나에게 Remote 저장소 수정권환이 있을경우
       - 내 branch를 다른 사람의 branch에 merge할 때 하기전에 검토해줘! 라고 하는것
    2. 나에게 수정권한이 없는 Remote 저장소 (주로 오픈소스)
       - 나 짱짱맨이니까 내 코드 한번 반영하는거 검토해줘! 놀라움을 선사해주지

# 수행방법
1. 원하는 Repository `fork`
2. Local에 git clone
3. 코드 수정이후 `branch` 만들어서 `fork한 내 github Repository`에 push
4. 아래 그림 버튼 클릭  
![](/images/2022-02-18-01-00-52.png)
5. comment 남기고 Create pull request 클릭

# 참고사이트
[참고사이트](https://www.youtube.com/watch?v=uvsz2XgRPfM)