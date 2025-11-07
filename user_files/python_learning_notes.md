# Python编程学习笔记

## 基础语法

Python是一种高级编程语言，具有以下特点：
- 语法简洁明了
- 面向对象编程
- 动态类型
- 解释型语言

## 数据类型

Python支持多种数据类型：

### 基础数据类型
- **整数 (int)**: 用于表示整数，如 `42`
- **浮点数 (float)**: 用于表示小数，如 `3.14`
- **字符串 (str)**: 用于表示文本，如 `"Hello World"`
- **布尔值 (bool)**: True 或 False

### 容器类型
- **列表 (list)**: 有序的可变序列，如 `[1, 2, 3]`
- **元组 (tuple)**: 有序的不可变序列，如 `(1, 2, 3)`
- **字典 (dict)**: 键值对集合，如 `{"name": "张三", "age": 25}`
- **集合 (set)**: 无序的唯一元素集合，如 `{1, 2, 3}`

## 控制结构

### 条件语句
```python
if condition:
    # 执行代码
elif another_condition:
    # 执行其他代码
else:
    # 默认执行代码
```

### 循环语句

#### for循环
```python
for item in sequence:
    # 对每个项目执行操作
```

#### while循环
```python
while condition:
    # 在条件为真时重复执行
```

## 函数定义

函数使用 `def` 关键字定义：

```python
def greet(name):
    return f"Hello, {name}!"

# 调用函数
result = greet("World")
print(result)  # 输出: Hello, World!
```

## 类和对象

Python是面向对象编程语言：

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"我是{self.name}，今年{self.age}岁"

# 创建对象
person = Person("张三", 25)
print(person.introduce())
```

## 异常处理

使用try-except处理异常：

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    print("清理代码")
```

## 模块和包

使用 `import` 导入模块：

```python
import math
from datetime import datetime

# 使用数学函数
result = math.sqrt(16)

# 获取当前时间
now = datetime.now()
```

## 实际应用示例

### 文件操作
```python
# 读取文件
with open("example.txt", "r", encoding="utf-8") as file:
    content = file.read()

# 写入文件
with open("output.txt", "w", encoding="utf-8") as file:
    file.write("Hello, World!")
```

### 网络请求
```python
import requests

# 发送HTTP请求
response = requests.get("https://api.example.com/data")
if response.status_code == 200:
    data = response.json()
```

## 学习建议

1. **多练习**: 通过编写小程序来巩固知识
2. **阅读文档**: 熟悉Python标准库
3. **参与项目**: 在实际项目中应用所学知识
4. **社区交流**: 加入Python社区，与其他开发者交流

Python是一门强大的语言，掌握基础知识后可以深入学习各种应用领域，如Web开发、数据科学、人工智能等。
