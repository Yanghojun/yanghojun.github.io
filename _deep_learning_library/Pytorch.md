---
layout: article
title:  "Pytorch"
categories: [딥러닝 라이브러리 모음집] # 홈페이지에서 카테고리를 통해 coding으로 지정되어있는 포스트들을 한번에 볼 수 있다
tag: [딥러닝, torch, torchvision, transforms, 전처리]
permalink: /Pytorch/
aside:
    toc: true
sidebar:
    nav: "study-nav"
---
| Pytorch 관련 데이터 전처리 방법, 시각화 방법들을 정리해두기 위한 포스트입니다.

# Torch.utils.data

Pytorch를 통해 데이터 셋을 구축하고자 하면 반드시 알아야 하는 부분이다.

<p align="center"> 
<img src="../images/20220509193348.png" width="70%" alt="전체 구조입니다.">
</p>
<div align="center">
전체적인 그림 구조 (출처: https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/)
</div>

데이터 형태가 index를 통해 접근 가능하거나, iterable 객체화가 가능한 데이터셋을 다음의  
1. `map-style datasets`
2. `iterable-style datasets`

라 하며 Pytorch는 위 2가지 타입을 다음과 같은 메소드를 통해 지원한다.

1. `map-style datasets`

    ```python
    __getitem__()

    __len__()
    ```

    - 데이터를 뽑을 때 torch.utils.data.Sampler를 통해 data를 loading 해야 한다.

2. `iterable-style datasets`

    ```python
    __iter__()
    ```

## Sampler

map-style datasets에 존재하는 여러 인덱스들을 sampler 객체를 활용해 iterable 객체로 만들 수 있다.
DataLoader 클래스에서 shuffle 인자를 True로 설정하면 자동으로 Sampler 객체가 생성되어 섞이는 것이다.

다음과 같은 Sampler 들이 존재한다.

- `SequentialSampler`: 항상 같은 순서
- `RandomSampler`: 랜덤, replacement 여부 선택 가능, 개수 선택 가능
- `SubsetRandomSampler`: 랜덤 리스트, 위와 두 조건 불가능
- `WeightRandomSampler`: 가중치에 따른 확률
- `BatchSampler`: Batch 단위로 Sampling 가능
- `DistributedSampler`: 분산처리(torch.nn.parallel.DistributedDataParallel과 함께 사용)


# torchvision.transforms

## torchvision.transforms.Compose

여러개의 전처리 작업을 한번에 묶기 위함

```python
>>> transforms.Compose([
>>>     transforms.CenterCrop(10),
>>>     transforms.PILToTensor(),
>>>     transforms.ConvertImageDtype(torch.float),
>>> ])
```


```python
import io

import requests
import torchvision.transforms as T

from PIL import Image

img = Image.open('./data/cat1.jpg')

width, height= img.size

preprocess = T.Compose([
   T.Resize(256),
   T.CenterCrop(50),
   T.ToTensor(),
#    T.Normalize(
#        mean=[0.485, 0.456, 0.406],
#        std=[0.229, 0.224, 0.225]
#    )
])

x = preprocess(img)

# Expected result
# torch.Size([3, 224, 224])
```

# 바운딩박스 좌표 읽어서 이미지 자르는 예시


```python
import json

def get_coor_from_json(path):
    with open(path) as f:
        raw_data = json.load(f)
        print(raw_data)

    return raw_data['shapes'][0]['points']
        

box_coor = get_coor_from_json('./cat1.json')    #[[x1, y1], [x2, y2]]

img = Image.open('./data/cat1.jpg')

x1,y1,x2,y2 = box_coor[0][0], box_coor[0][1], box_coor[1][0], box_coor[1][1]
print(x1, y1, x2, y2)
img = img.crop([x1, y1, x2, y2])
img.show()
```

    {'version': '5.0.1', 'flags': {}, 'shapes': [{'label': 'normal', 'points': [[171.99999999999997, 46.666666666666664], [333.8181818181818, 172.72727272727272]], 'group_id': None, 'shape_type': 'rectangle', 'flags': {}}], 'imagePath': 'data\\cat1.jpg', 'imageData': None, 'imageHeight': 408, 'imageWidth': 612}
    171.99999999999997 46.666666666666664 333.8181818181818 172.72727272727272
    


```python
img = Image.open('./data/cat1.jpg')

width, height = img.size

img = img.crop([0,0,width/2,height/2])
img.show()
```