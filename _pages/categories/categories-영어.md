---
title: "영어"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/영어
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"     # Navigation에 있는 docs
---

> 컴퓨터 공부와 영어는 떼어놓을 수 없는 존재이기에.. 이것은 Computer Language에 들어가야 할 영역일지어다...

{% assign posts = site.categories.["영어"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} 
<!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->