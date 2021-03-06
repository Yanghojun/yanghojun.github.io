---
layout: article
title:  "Matplotlib"
category: [시각화]
permalink: /Matplotlib/
tag: [시각화, matplotlib, python]
aside:
    toc: true
sidebar:
    nav: "study-nav"
---
# Matplotlib
## 기초


```python
import matplotlib.pyplot as plt

plt.plot([2, 3, 5, 10])     # list, tuple, numpy 같은 Array 입력받을 수 있음
plt.show()  # x 값은 기본적으로 [0, 1, 2, 3]이 됨
```
    
![png](/images/output_1_0.png)
    

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.show()
```


    
![png](/images/output_2_0.png)
    



```python
import matplotlib.pyplot as plt

data_dict = {'data_x': [1, 2, 3, 4, 5], 'data_y': [2, 3, 5, 10, 8]}

plt.plot('data_x', 'data_y', data=data_dict)        # dictionary와 연계가능. 먼저 dictionary를 제공하고 그것의 Key를 호출한다 생각하면 편함
plt.show()
```


    
![png](/images/output_3_0.png)
    


## 축 글씨 및 길이 지정

### 레이블 여백, 폰트 설정, 위치 설정


```python
import matplotlib.pyplot as plt

font1 = {'family': 'serif',
         'color': 'b',
         'weight': 'bold',
         'size': 14
         }

font2 = {'family': 'fantasy',
         'color': 'deeppink',
         'weight': 'normal',
         'size': 'xx-large'
         }

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis', labelpad=15, fontdict=font1, loc='right')
plt.ylabel('Y-Axis', labelpad=20, fontdict=font2, loc='top')
plt.show()
```


    
![png](/images/output_6_0.png)
    


## 여러 그래프 표시
- plt.subplot, plt.subplots 2가지 방식이 있음
- 구조도 이해  
![](/images/subplot_03.png)

### plt.subplot


```python
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1 / 행렬 모양을 지정하고, 인덱스 지정한다 생각하면 되겠네
plt.plot(x1, y1, 'o-')
plt.title('1st Graph')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
plt.plot(x2, y2, '.-')
plt.title('2nd Graph')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.tight_layout()
plt.show()
```


    
![png](/images/output_8_0.png)
    

```python
fig , (ax1, ax2, ax3) = plt.subplots(figsize=(14,4), ncols=3)   # 이런식으로 써서도 접근 가능하다
```

```python
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(1, 2, 1)                # nrows=1, ncols=2, index=1
plt.plot(x1, y1, 'o-')
plt.title('1st Graph')
plt.xlabel('time (s)')
plt.ylabel('Damped oscillation')

plt.subplot(1, 2, 2)                # nrows=1, ncols=2, index=2
plt.plot(x2, y2, '.-')
plt.title('2nd Graph')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.tight_layout()
plt.show()
```


    
![png](/images/output_9_0.png)
    


### plt.subplots


```python
fig, axes = plt.subplots(2,2)       # fig: picture를 담은 객체, axes: 각각의 그래프를 Array 형태로 담음
axes[0][0].plot([1,2,3])
axes[1][0].plot([4,5,4,5])
```




    [<matplotlib.lines.Line2D at 0x1edc5b10460>]




    
![png](/images/output_11_1.png)
    


## 마커 지정하기
- 지정할 수 있는 것들

![](/images/2022-01-07-02-02-46.png)


```python
import matplotlib.pyplot as plt

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo-')    # 파란색 + 마커(o는 동그란거) + 실선
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo--')     # 파란색 + 마커(o는 동그란거) + 점선
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
```


    
![png](/images/output_13_0.png)

## scatter를 활용해 군집화 데이터 표현하기
```python
import matplotlib.pyplot as plt
%matplotlib inline

clusterDF['meanshift_label']  = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers=['o', 's', '^', 'x', '*']

for label in unique_labels:
    label_cluster = clusterDF[clusterDF['meanshift_label']==label]
    center_x_y = centers[label]
    # 군집별로 다른 마커로 산점도 적용
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], edgecolor='k', marker=markers[label] )
    
    # 군집별 중심 표현
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='gray', alpha=0.9, marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k', marker='$%d$' % label)
    
plt.show()
```
![](/images/2022-02-13-15-58-58.png)

# Seaborn

## 간단한 2개 칼럼 시각화

```python
sns.barplot(x='Sex', y = 'Survived', data=titanic_df)  # DataFrame과 연계하여 표현 가능하다. x, y는 DataFrame의 칼럼명을 넣어주면 된다
```
![](/images/2022-01-05-17-43-39.png)

## 3개 칼럼 시각화. (Y축은 Survived로 고정)

- hue를 통해서 bar(막대기)를 어떻게 세세하게 나눌것이냐는 것을 결정함
    - 막대기는 x 칼럼, y 칼럼의 정보가 모두 들어가 있는것임. 이 막대기에 조건을 어떻게 붙일것인가를 hue가 결정

```python
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
```

![](/images/2022-01-05-17-43-54.png)

## Age 같이 굉장히 많은 데이터 종류가 있을 시 카테고리 형태로 변환해서 사용 가능

```python
# 입력 age에 따라 구분값을 반환하는 함수 설정. DataFrame의 apply lambda식에 사용. 
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

# 막대그래프의 크기 figure를 더 크게 설정 
plt.figure(figsize=(10,6))

#X축의 값을 순차적으로 표시하기 위한 설정 
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# lambda 식에 위에서 생성한 get_category( ) 함수를 반환값으로 지정. 
# get_category(X)는 입력값으로 'Age' 컬럼값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y = 'Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)
```

# Matplot

## precision_recall_curve 그리기

```python
from sklearn.metrics import precision_recall_curve

# 레이블 값이 1일때의 예측 확률을 추출 
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1] 

# 실제값 데이터 셋과 레이블 값이 1일 때의 예측 확률을 precision_recall_curve 인자로 입력 
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1 )
print('반환된 분류 결정 임곗값 배열의 Shape:', thresholds.shape)
print('반환된 precisions 배열의 Shape:', precisions.shape)
print('반환된 recalls 배열의 Shape:', recalls.shape)

print("thresholds 5 sample:", thresholds[:5])
print("precisions 5 sample:", precisions[:5])
print("recalls 5 sample:", recalls[:5])

#반환된 임계값 배열 로우가 147건이므로 샘플로 10건만 추출하되, 임곗값을 15 Step으로 추출. 
thr_index = np.arange(0, thresholds.shape[0], 15)
print('샘플 추출을 위한 임계값 배열의 index 10개:', thr_index)
print('샘플용 10개의 임곗값: ', np.round(thresholds[thr_index], 2))

# 15 step 단위로 추출된 임계값에 따른 정밀도와 재현율 값 
print('샘플 임계값별 정밀도: ', np.round(precisions[thr_index], 3))
print('샘플 임계값별 재현율: ', np.round(recalls[thr_index], 3))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value') 
    plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )
```

![](/images/2022-01-05-17-44-08.png)

# Boolean indexing 에러 날 때
- DataFrame의 index값이 올바르게 되어있는지 check
  - 가령 데이터가 100개이고, label은 0, 1인 numpy로 이루어져 있다고 하자. 이때 데이터의 인덱스가 150 같은 숫자가 들어가있다면 boolean indexing 할 때 에러남

# 데이터 분포형태 살펴보기
```python
DataFrame.hist()  # target 데이터 같은게 들어있는 DataFrame이라고 가정
sns.distplot(DataFrame)  # 이걸로도 가능. (이게 더 이쁜듯)
```
![](/images/2022-01-19-19-42-49.png)    

- hist()처럼 아무 인자도 주지 않았을 경우 x축이 DataFrame 값임(즉 위 그림에선 0 ~ 200인 값이 대략 4200개 있다는 것)
- 위 그림은 데이터가 초반부에 치우쳐 있는 모습