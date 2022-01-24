---
title: "Paper-Review"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/Paper-Review
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.["Paper-Review"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} <!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->