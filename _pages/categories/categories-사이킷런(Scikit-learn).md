---
title: "사이킷런(Scikit-learn)"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/사이킷런(Scikit-learn)
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"     # Navigation에 있는 docs
---
> 사이킷런 라이브러리를 다루는 중 이해가 어렵거나 쓰기 쉽게 정리하기 위함
{% assign posts = site.categories.["사이킷런(Scikit-learn)"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} 
<!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->