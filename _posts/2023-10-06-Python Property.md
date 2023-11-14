---
layout: article
title:  "Python Property"
category: [Computer Language]
tag: [Python, 파이썬 문법]
permalink: /PythonProperty}/
show_author_profile: true
aside:
    toc: true
sidebar:
    nav: "study-nav"
---

# Property?

Python 내부 함수

# 이거 왜 쓸까?

클래스 Attribute를 **쓰는 입장에서 변화를 못느끼지만**, setter, getter 로직을 내부적으로 변경할 수 있음.
즉, 사용 측면에서의 하위 호환성을 유지하면서 기능 수정이 가능하다는 것.

구현측면에선 보편적으로 property는 Attribute에 값을 대입하거나(setter) Attribute 값을 받을 때(getter) 그 값의 범위 등을 분기해서 에러를 처리하는 등의 로직에 주로 사용한다.

# 코드 및 설명


```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
hojun = Person("Hojun", 20)
print(hojun.age)

hojun.age = 25
print(hojun.age)

hojun.age = -1
print(hojun.age)
```

    20
    25
    -1
    

Age가 -1인것은 말이 안되므로 이를 예외처리 해준다고 하자.  
그러기 위해선 `setter 함수`를 하나 추가해줘야 한다. 모양 맞게 `getter 함수`도 추가해주자.


```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.set_age(age)
    
    def set_age(self, age):
        if age < 0:
            raise ValueError('Invalid age')
        self._age = age         # _age를 사용했다. Python 코딩컨벤션으로, 
                                # _를 하나 붙인 변수는 외부에서 직접 접근해서 쓰지 말것을 의미한다.
    
    def get_age(self):
        return self._age
    
hojun = Person("Hojun", 20)
print(hojun.get_age())

hojun.set_age(25)
print(hojun.get_age())

hojun.set_age(-1)
print(hojun.get_age())
```

    20
    25
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[19], line 21
         18 hojun.set_age(25)
         19 print(hojun.get_age())
    ---> 21 hojun.set_age(-1)
         22 print(hojun.get_age())
    

    Cell In[19], line 8, in Person.set_age(self, age)
          6 def set_age(self, age):
          7     if age < 0:
    ----> 8         raise ValueError('Invalid age')
          9     self._age = age
    

    ValueError: Invalid age


원했던 기능 추가는 됐지만, Person 클래스의 age Attribute를 사용하는 방식이 달라진다.  

- age 설정할 때
  - 기존: hojun.age = num, 현재: hojun.set_age(num)
- age 출력할 때
  - 기존: hojun.age, 현재: hojun.get_age(num)

협업과정에서 다른 사람이 Person 클래스의 age Attribute를 쓰고 있었다면 문제가 발생하게 된다.

그래서 본 포스트에서 맨 처음에 말한 문구인 **쓰는 입장에서 변화를 못느끼게 하는것** 이 중요하다.

이는 Python 내부 함수인 `property`를 통해 가능하다.


```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.set_age(age)
    
    def set_age(self, age):
        if age < 0:
            raise ValueError('Invalid age')
        self._age = age         # _age를 사용했다. Python 코딩컨벤션으로, 
                                # _를 하나 붙인 변수는 외부에서 직접 접근해서 쓰지 말것을 의미한다.
    
    def get_age(self):
        return self._age
    
    age = property(get_age, set_age)
    
hojun = Person("Hojun", 20)
print(hojun.age)

hojun.age = 25
print(hojun.age)

hojun.age = -1
print(hojun.age)
```

    20
    25
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[23], line 23
         20 hojun.age = 25
         21 print(hojun.age)
    ---> 23 hojun.age = -1
         24 print(hojun.age)
    

    Cell In[23], line 8, in Person.set_age(self, age)
          6 def set_age(self, age):
          7     if age < 0:
    ----> 8         raise ValueError('Invalid age')
          9     self._age = age
    

    ValueError: Invalid age


age Attribute를 **직접 접근해서 쓰는 방식으로 다시 변경했음**을 주목하자.  
즉, 사용 측면에서 클래스 하위호환성을 깨지 않고 age가 음수일 경우 Error를 발생시키는 기능을 추가했다.

여기서 `Decorator` 기능을 통해 더 쌈빡하게 코드를 줄일 수 있다.


```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self._age = age         # _age를 사용했다. Python 코딩컨벤션으로, 
                                # _를 하나 붙인 변수는 외부에서 직접 접근해서 쓰지 말것을 의미한다.
    
    # def set_age(self, age):
    #     if age < 0:
    #         raise ValueError('Invalid age')
    #     self._age = age
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, age):
        if age<0:
            raise ValueError('Invalid age')
        self._age = age
    
    # age = property(get_age, set_age)
    
hojun = Person("Hojun", 20)
print(hojun.age)

hojun.age = 25
print(hojun.age)

hojun.age = -1
print(hojun.age)
```

    20
    25
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[28], line 29
         26 hojun.age = 25
         27 print(hojun.age)
    ---> 29 hojun.age = -1
         30 print(hojun.age)
    

    Cell In[28], line 18, in Person.age(self, age)
         15 @age.setter
         16 def age(self, age):
         17     if age<0:
    ---> 18         raise ValueError('Invalid age')
         19     self._age = age
    

    ValueError: Invalid age

