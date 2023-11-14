---
layout: article
title:  "Python Decorator"
category: [Computer Language]
tag: [Python, 파이썬 문법]
permalink: /PythonDecorator/
show_author_profile: true
aside:
    toc: true
sidebar:
    nav: "study-nav"
---

# Decorator?

- @ 함수명, 클래스명 앞에 붙인거

# 이거 왜 쓸까?

- 함수, 클래스를 수정하지 않고 새로운 기능을 넣어줄 수 있음 (그래서 Decorator - 장식)
  - 클래스를 사용해서 Decorator 역할을 비슷하게 수행 할 수 있기는 함.

    ```python
    class base:
        def __init__(self, name):
            self.name = name

        def say(self):
            print("Hi")
            print(f"My name is {self.name}")
            print("Bye")

    hojun = base("hojun")
    jee = base("Jee")

    hojun.say()
    jee.say()
    ```

    그러나, `say` 같은 함수가 이미 선언된 상태였다면, 클래스화를 시켜준다거나, 이미 메서드라면 메서드 내부 코드를 직접 건드려야함. 이러지 말자는게 Decorator의 의도로 보임.

  - 클래스 상속 기능등을 이용해서 Decorator 역할 할 수 있을 것 같은데, 굳이굳이 쓰는 이유는 **Design pattern** 이라고 함


```python
def add_author(func):       # decorator
    print('Author: Natalia Tsarkova')
    return func

@add_author
def decorators_article():   # target function to be decorated
    print('Article: Decorators in Python')

decorators_article()
```

    Author: Natalia Tsarkova
    Article: Decorators in Python
    

1. add_author 함수가 수정할 예정인 함수를 인자로 받는다. (위 코드 예시에선 `decorators_article 함수`)
2. 필요한 기능을 추가한 뒤, 인자로 받았던 함수를 반환한다.

이를 통해, `decorators_article` 함수 내부를 수정하지 않고, Author를 출력하는 기능을 추가했다.  
아래 코드는 완벽히 같은 기능을 수행한다. 아래 처럼 쓰는것보다 `@` 붙여서 쓰는게 더 간편한 것임을 보여준다.


```python
def add_author(func):       # decorator
    print('Author: Natalia Tsarkova')
    return func

def decorators_article():   # target function to be decorated
    print('Article: Decorators in Python')

decorators_article = add_author(decorators_article)
decorators_article()
```

    Author: Natalia Tsarkova
    Article: Decorators in Python
    

target function의 수행시간을 측정하거나, logging 등을 수행하기 위해 데코레이터 내에서 호출해야 한다면  
`wrapper()` 함수를 내부에 추가로 선언해서 아래와 같이 사용하자.


```python
import time
def time_of_wailting_for_snail(func_slow):
    def wrapper():
        start = time.time()
        func_slow()
        end = time.time()
        print('Snail was slow for: {} seconds'.format(end - start))     
    return wrapper    
@time_of_wailting_for_snail
def snail_greets_world():
    time.sleep(3)
    print("Hi, World! 🐌")
snail_greets_world()
```

    Hi, World! 🐌
    Snail was slow for: 3.0024163722991943 seconds
    
