---
title: "기초수학"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/기초수학
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"     # Navigation에 있는 docs
---

> 머리가 좋지 못해 수학 이해가 버거운 Juneer에게 힘이 되어줄 카테고리..

{% assign posts = site.categories.["기초수학"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} 
<!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->