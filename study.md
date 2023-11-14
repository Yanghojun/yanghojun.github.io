---
layout: article
title: 공부방
sidebar:
  nav: study-nav
---



<div align="center" markdown="1"> 
## 데스크탑 환경의 경우: 좌측의 메뉴로 원하는 카테고리를 참고하세용~
## 모바일 환경의 경우: 플로팅 메뉴버튼을 클릭해 원하는 카테고리를 참고하세용~
</div>

<!-- Include the library. -->
<script
  src="https://unpkg.com/github-calendar@latest/dist/github-calendar.min.js"
></script>

<!-- Optionally, include the theme (if you don't want to struggle to write the CSS) -->
<link
   rel="stylesheet"
   href="https://unpkg.com/github-calendar@latest/dist/github-calendar-responsive.css"
/>

<div>
    <!-- Prepare a container for your calendar. -->
    <div style="text-align: center;"><strong>내 Github contribution (매일매일 성장하자)</strong></div> 
    <div class="calendar">
        <!-- Loading stuff -->
        잔디 심는중...
    </div>
</div>

<script>
    GitHubCalendar(".calendar", "yanghojun", { responsive: true, tooltips: true, global_stats: true}).then(function() {
        // delete the space underneath the module bar which is caused by minheight 
        document.getElementsByClassName('calendar')[0].style.minHeight = "100px";
        // hide more and less legen below the contribution graph
        document.getElementsByClassName('contrib-legend')[0].style.display = "none";
    });
</script>

