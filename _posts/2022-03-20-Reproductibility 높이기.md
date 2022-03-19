---
layout: single
title:  "Reproductibility 높이기"
categories: [Pytorch] # 홈페이지에서 카테고리를 통해 coding으로 지정되어있는 포스트들을 한번에 볼 수 있다
tag: [Pytorch, Reproductibility, 재현성]
permalink: /Reproductibility 높이기/
toc: true
toc_sticky: true
author_profile: false # 왼쪽에 조그마한 글 나오는지 여부
sidebar:
    nav: "docs"     # Navigation에 있는 docs
# search: false # 만약 이 포스트가 검색이 안되길 원한다면 주석해제
---

> 학습을 진행할 때 마다 결과가 달리질 수 있으며, 항상 동일한 결과가 나오기를 원할 경우 Reproductibility(재현성)을 높여야함. 아래 코드를 통해 재현성을 높일 수 있음

```python
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
```