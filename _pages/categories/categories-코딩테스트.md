---
title: "코딩테스트"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/코딩테스트
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"     # Navigation에 있는 docs
---

{% assign posts = site.categories.["코딩테스트"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} <!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->