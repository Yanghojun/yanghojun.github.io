---
title: "객체지향 프로그래밍"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/객체지향 프로그래밍
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.["객체지향 프로그래밍"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} <!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->

