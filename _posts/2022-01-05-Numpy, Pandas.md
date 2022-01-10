---
layout: single
title:  "Numpy, Pandas"
categories: [Machine-Learning] # 홈페이지에서 카테고리를 통해 coding으로 지정되어있는 포스트들을 한번에 볼 수 있다
tag: [Python, Pandas, DataFrame, Numpy]
permalink: /Numpy, Pandas/
toc: true
author_profile: false # 왼쪽에 조그마한 글 나오는지 여부
sidebar:
    nav: "docs"     # Navigation에 있는 docs
# search: false # 만약 이 포스트가 검색이 안되길 원한다면 주석해제
---

# Series, DataFrame

- Series는 칼럼이 한개 / DataFrame은 칼럼이 여러개

# Series

- Key, value 형태로 이루어져 있음. Dict타입과 호환 가능
- 리스트와도 호환가능. 이때는 정수형 인덱스가 들어감
- 인덱스 종류
    1. 정수형 인덱스
        
        ```python
        sr[0], sr[0:2]
        ```
        
    2. 인덱스 이름 
        
        ```python
        sr['c'], sr['c': 'e']
        ```
        

# DataFrame

- 각 행이 'record'라고 불리며, 각 열이 'feature'라고 불림


## 기능
    
  - 정보 조회
      
      ```python
      DataFrame.info()
      
      DataFrame.describe()  # 숫자형 칼럼(int, float)에 대해서만 계산하며 object는 자동으로 제외시킨다
      
      DataFrame['column_name'].value_counts()  # Series로 반환시킨 데이터에 대해 특정 값이 몇번 나오는지를 알 수 있게 해줌. 즉 데이터 분포도 파악이 가능한 좋은 함수
      ```
      
      ![](/images/2022-01-05-20-38-23.png)
      
      ![](/images/2022-01-05-20-38-30.png)
      
      ![](/images/2022-01-05-20-38-36.png)
      
  - DataFrame → Dict 형태로의 변환
      
      ```python
      # DataFrame을 딕셔너리로 변환
      dict3 = df_dict.to_dict('list')    # 이런식으로 'list'를 인자로 주면 훨씬 깔끔하다
      print('\n df_dict.to_dict() 타입:', type(dict3))
      print(dict3)
      
      dict3 = df_dict.to_dict()
      print('\n df_dict.to_dict() 타입:', type(dict3))
      print(dict3)
      
      '''output
      df_dict.to_dict() 타입: <class 'dict'>
      {'col1': [1, 11], 'col2': [2, 22], 'col3': [3, 33]}
      
       df_dict.to_dict() 타입: <class 'dict'>
      {'col1': {0: 1, 1: 11}, 'col2': {0: 2, 1: 22}, 'col3': {0: 3, 1: 33}}
      '''
      ```
      
### 인덱싱
  - Numpy와 달리 [ ]에 0, 1과 같은 값을 넣을 경우 에러 발생. 칼럼명을 넣어줘야함
      
      ```python
      print('단일 컬럼 데이터 추출:\n', titanic_df[ 'Pclass' ].head(3))
      print('\n여러 컬럼들의 데이터 추출:\n', titanic_df[ ['Survived', 'Pclass'] ].head(3))
      print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])
      
      '''output
      단일 컬럼 데이터 추출:
       0    3
      1    1
      2    3
      Name: Pclass, dtype: int64
      
      여러 컬럼들의 데이터 추출:
          Survived  Pclass
      0         0       3
      1         1       1
      2         1       3
      
      KeyError: 0
      '''
      ```
      
  - 슬라이싱은 가능
      
      ```python
      titanic_df[0:2]
      
      '''output
      PassengerId  Survived  Pclass            Name     Sex   Age  SibSp  Parch     Ticket     Fare Cabin Embarked
      0            1         0       3  Braund, Mr....    male  22.0      1      0  A/5 21171   7.2500   NaN        S
      1            2         1       1  Cumings, Mr...  female  38.0      1      0   PC 17599  71.2833   C85        C
      '''
      ```
      
  - 인덱싱 방식은 위치(Position)기반 인덱싱, 명칭(Label)기반 인덱싱이 있으며 그냥 DataFrame[ ]과 같은 형식으로 데이터 접근이 가능하지만 내 생각에 위치기반 인덱싱, 명칭 기반 인덱싱을 `명확히` 구분하게 하기 위해서 iloc, loc 방식을 사용하는것으로 보임
      - iloc (Integer - location): 위치 기반 인덱싱
          - 0 부터 시작하는 행, 열의 위치 좌표에만 의존함
          
          ```python
          data_df_reset.iloc[0, 1]
          
          '''output
          'Chulmin'
          '''
          ```
          
      - loc (label - location): 명칭 기반 인덱싱
          - 인덱스, 칼럼명으로 접근 ⇒ 행 위치에 `DataFrame 인덱스`, 열 위치에 `칼럼명`
              - 행 위치같은 경우 불린 인덱싱 지원을 위해 시리즈타입은 호환이 됨 (간단 테스트로 정수형 Series 넣으니까 접근 됐음. 아마 DataFrame.index를 정수가 아니라 문자열 같은걸로 변경하고 다시 실행하면 문자형은 못찾음)
          - 명칭 기반 인덱싱은 DataFrame.index 객체 (정확히는 pandas.core.indexes.range.RangeIndex)에서 값을 찾음
          - 명칭 기반 인덱싱은 슬라이싱 할 때 종료 값을 포함한 범위가 반환됨. 위치기반 인덱싱과 혼돈될 수 있는 부분이여서 `매우 조심해야함`
          - 행 위치에 DataFrame
          
          ```python
          data_df.loc['one', 'Name']
          
          data_df_reset.loc[1, 'Name'] # 이런식으로 index가 정수형일 경우엔 정수 입력 가능. 명칭 기반이라고 해서 숫자만 써야하는것 아님
          													# data_df_reset 데이터프레임은 reset_index를 통해 1부터 인덱스가 시작되도록 변경해놓은 데이터프레임이여서 첫번째 줄의 Name 칼럼 값이 나옴
          
          '''output
          'Chulmin'
          
          'Chulmin'
          '''
          
          print('명칭기반 ix slicing\n', data_df.ix['one':'two', 'Name'],'\n')
          print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0],'\n')
          print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name'])
          
          '''output
          명칭기반 ix slicing
           one     Chulmin
          two    Eunkyung
          Name: Name, dtype: object 
          
          위치기반 iloc slicing
           one    Chulmin
          Name: Name, dtype: object 
          
          명칭기반 loc slicing
           one     Chulmin
          two    Eunkyung
          Name: Name, dtype: object
          '''
          ```
          
      - **불린 인덱싱 (중요)**
          - loc로 생각해보면 인덱스, 칼럼명을 입력해줘야 하는데 칼럼명은 생략했으니 모든 칼럼을 출력, 인덱스의 경우 Boolean type을 가지고 있는 Series도 호환이 가능하다
          - iloc은 불린 인덱싱 지원 X
          - and, or, not 조건 연산자를 활용해서 응용 가능
          
          ```python
          hj_list = []
          for i in range(891):
              if i == 460:
                  hj_list.append(True)
              else:
                  hj_list.append(False)
          
          hj_list = np.array(hj_list)
          hj_list = pd.Series(hj_list)
          
          titanic_df[hj_list], titanic_df.loc[hj_list]
          
          '''output
          PassengerId  Survived  Pclass            Name   Sex   Age  SibSp  Parch Ticket   Fare Cabin Embarked
          460          461         1       1  Anderson, M...  male  48.0      0      0  19952  26.55   E12        S
          
          PassengerId  Survived  Pclass            Name   Sex   Age  SibSp  Parch Ticket   Fare Cabin Embarked
          460          461         1       1  Anderson, M...  male  48.0      0      0  19952  26.55   E12        S
          '''
          
          titanic_df[ (titanic_df['Age'] > 60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')]
          
          '''output
          PassengerId  Survived  Pclass            Name     Sex   Age  SibSp  Parch  Ticket     Fare Cabin Embarked
          275          276         1       1  Andrews, Mi...  female  63.0      1      0   13502  77.9583    D7        S
          829          830         1       1  Stone, Mrs....  female  62.0      0      0  113572  80.0000   B28      NaN
          '''
          ```
                
### groupby( ) 적용
  - by 인자에 지정한 칼럼명을 기준으로 삼음
  - Aggregation 함수와 연계 가능하며 이중 count( )를 사용한것을 예로들면 Pclass가 1일때 각 칼럼별로 몇개씩 있는건지 테이블로 나옴
      
      ```python
      titanic_groupby = titanic_df.groupby(by='Pclass')
      titanic_groupby.count()
      
      '''output
      PassengerId  Survived  Name  Sex  Age  SibSp  Parch  Ticket  Fare  Cabin  Embarked
      Pclass                                                                                    
      1               216       216   216  216  186    216    216     216   216    176       214
      2               184       184   184  184  173    184    184     184   184     16       184
      3               491       491   491  491  355    491    491     491   491     12       491
      '''
      ```
      
  - 여러개의 Aggregation과 연계 가능하며 .agg( ) 를 사용해야함
      
      ```python
      titanic_df.groupby('Pclass')['Age'].agg([max, min])
      
      '''output
                      max   min
      Pclass            
      1       80.0  0.92
      2       70.0  0.67
      3       74.0  0.42
      '''
      ```
      
  - 여러개의 칼럼에 대해 여러개의 Aggregation을 적용하려면 Dict 형태로 사용한다
      
      ```python
      agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
      titanic_df.groupby('Pclass').agg(agg_format)
      
      '''output
                      Age  SibSp       Fare
      Pclass                        
      1       80.0     90  84.154687
      2       70.0     74  20.662183
      3       74.0    302  13.675550
      '''
      ```
            
### 조건에 따라서 값 변경시키기
- apply lambda식 사용
    - lambda란?
        - 함수의 선언과 함수내의 처리를 한 줄의 식으로 쉽게 변환하는 것
        
        ![](/images/2022-01-05-20-38-48.png)
        
        ```python
        def get_square(a):
            return a**2
        
        print('3의 제곱은:',get_square(3))
        
        '''output
        3의 제곱은: 9
        '''
        
        ## 람다 적용 후
        lambda_square = lambda x : x ** 2
        print('3의 제곱은:',lambda_square(3))
        
        '''output
        3의 제곱은: 9
        '''
        ```
        
        - 입력 인자를 하나가 아니라 여러개로 받고 싶다면? `map( )` 함수 사용
            - map( )은 리스트의 요소들을 지정한 함수로 처리할 수 있게끔 해줌
            - 1, 2, 3을 입력인자로 받아서 지정한 함수(여기서는 lambda x: x**2) 로 처리
                
                ```python
                a=[1,2,3]
                squares = map(lambda x : x**2, a)
                list(squares)
                ```
                
    - DataFrame에서는 map이 아닌 apply를 사용 (로직은 거의 흡사함)
        - 다른 부분은 map('함수', '리스트' ) 에서 '리스트'가 입력인자로 지정되는 것이였다면 apply에는 따로 입력인자로 지정할 것이 없음. DataFrame의 칼럼값이 입력인자로 바로 들어감
            
            ```python
            titanic_df['Name_len']= titanic_df['Name'].apply(lambda x : len(x))
            titanic_df[['Name','Name_len']].head(3)
            
            '''output
                                    Name  Name_len
            0  Braund, Mr....        23
            1  Cumings, Mr...        51
            2  Heikkinen, ...        22
            '''
            ```
            
            ![](/images/2022-01-05-20-39-04.png)

        - lambda는 if, else를 지원
            - lambda 식 ' : ' 의 오른편에는 반환 값이 있어야 하기 때문에 조금 독특한 구조를 띔
            - else if를 지원하지 않기 때문에 붙는 조건이 많다면 아예 따로 함수를 지정해주는게 좋음
                
                ```python
                titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult' )
                titanic_df[['Age','Child_Adult']].head(8)
                
                '''output
                            Age Child_Adult
                
                0  22.000000       Adult
                1  38.000000       Adult
                2  26.000000       Adult
                3  35.000000       Adult
                4  35.000000       Adult
                5  29.699118       Adult
                6  54.000000       Adult
                7   2.000000       Child
                '''
                
                titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x <= 60 else    # ( ) 열고 else if 를 처리한 모습
                                                                                                    'Elderly'))
                titanic_df['Age_cat'].value_counts()
                
                '''output
                Adult      786
                Child       83
                Elderly     22
                Name: Age_cat, dtype: int64
                '''
                
                # 나이에 따라 세분화된 분류를 수행하는 함수 생성. 
                def get_category(age):
                    cat = ''
                    if age <= 5: cat = 'Baby'
                    elif age <= 12: cat = 'Child'
                    elif age <= 18: cat = 'Teenager'
                    elif age <= 25: cat = 'Student'
                    elif age <= 35: cat = 'Young Adult'
                    elif age <= 60: cat = 'Adult'
                    else : cat = 'Elderly'
                    
                    return cat
                
                # lambda 식에 위에서 생성한 get_category( ) 함수를 반환값으로 지정. 
                # get_category(X)는 입력값으로 ‘Age’ 컬럼 값을 받아서 해당하는 cat 반환
                titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
                titanic_df[['Age','Age_cat']].head()
                
                '''output
                Age      Age_cat
                0  22.0      Student
                1  38.0        Adult
                2  26.0  Young Adult
                3  35.0  Young Adult
                4  35.0  Young Adult
                '''
                ```