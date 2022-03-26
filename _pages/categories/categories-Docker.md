---
title: "Docker"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/Docker
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"     # Navigation에 있는 docs
---

> 본 카테고리는 WSL2(Windows 11) + Vscode + Docker 환경에서 효율적으로 딥러닝 프로젝트를 진행하기 위함입니다.
{% assign posts = site.categories.["Docker"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} 
<!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->