---
title: "통계학"
layout: archive     # tag 눌렀을 때 나오는 레이아웃 형식
permalink: categories/통계학
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"     # Navigation에 있는 docs
---

> [친절한 데이터 사이언티스트 되기 강좌](https://recipesds.tistory.com/)를 통해 통계 기초를 잡아보자!

<object data="http://yanghojun.github.io/pdf/cs231n_Recurrent_Neural_Network.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yanghojun.github.io/pdf/cs231n_Recurrent_Neural_Network.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yanghojun.github.io/pdf/cs231n_Recurrent_Neural_Network.pdf">Download PDF</a>.</p>
    </embed>
</object>

{% assign posts = site.categories.["통계학"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %} 
<!-- archive-single.html에서 같은 카테고리, 태그를 모아두는 역할을 함 -->