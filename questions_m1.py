"""
questions_m1.py — Module 1: Python Core
3 Practice sets + 1 Final assessment
Topics: Data types, control flow, data structures, functions, OOP, error handling
"""

import hashlib, re

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 1 — Data Types + Data Structures
# ════════════════════════════════════════════════════════════════
M1_PRACTICE_1 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"What is the output of:\n\ntype(1/2) == type(1//2)",
     "opts":["True","False","TypeError","None"],
     "ah":h("False")},
    {"id":2,"type":"mcq","marks":2,
     "text":"Which of the following will raise a TypeError?",
     "opts":["'5' + '3'","'5' * 3","'5' + 3","str(5) + '3'"],
     "ah":h("'5' + 3")},
    {"id":3,"type":"fill","marks":3,
     "text":"What is the output of:\n\n[x for x in range(10) if x%2==0 and x%3==0]\n\nType exactly as Python would print it.",
     "ah":h("[0, 6]")},
    {"id":4,"type":"fill","marks":3,
     "text":"What is the output of:\n\n{k:v for k,v in zip('abcde', range(5))}\n\nType exactly as Python would print it.",
     "ah":h("{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}")},
    {"id":5,"type":"fill","marks":3,
     "text":"Given:\n\nd = {'x':10, 'y':20, 'z':30}\n\nWhat does this return?\n\n{k:v for k,v in d.items() if v > 15}",
     "ah":h("{'y': 20, 'z': 30}")},
    {"id":6,"type":"fill","marks":4,
     "text":"What is the output of:\n\nlist(map(lambda x: x**3, filter(lambda x: x%2==0, range(1,6))))\n\nType exactly as Python would print it.",
     "ah":h("[8, 64]")},
    {"id":7,"type":"code","marks":4,
     "text":"What is the output of this closure?\n\ndef outer(x):\n    def inner(y): return x + y\n    return inner\nadd5 = outer(5)\nprint(add5(3))",
     "preload":"","exp":"8"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:\n  data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\n\nWrite code to return a dict where keys are unique elements\nand values are their counts. Print sorted by key.",
     "preload":"data = [3,1,4,1,5,9,2,6,5,3,5]\n",
     "exp":"{1: 2, 2: 1, 3: 2, 4: 1, 5: 3, 6: 1, 9: 1}"},
    {"id":9,"type":"code","marks":5,
     "text":"Write a function flatten(lst) that flattens a nested list of any depth.\n\nPreloaded:\n  nested = [1, [2, 3], [4, [5, 6]], 7]\n\nPrint the flattened result.",
     "preload":"nested = [1, [2, 3], [4, [5, 6]], 7]\n",
     "exp":"[1, 2, 3, 4, 5, 6, 7]"},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 2 — Control Flow + Functions
# ════════════════════════════════════════════════════════════════
M1_PRACTICE_2 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"What does this print?\n\nfor i in range(3):\n    pass\nelse:\n    print('done')",
     "opts":["done","nothing","error","0 1 2 done"],
     "ah":h("done")},
    {"id":2,"type":"mcq","marks":2,
     "text":"What is the output of:\n\nx = 5\nresult = 'high' if x > 3 else 'low' if x > 1 else 'zero'\nprint(result)",
     "opts":["high","low","zero","error"],
     "ah":h("high")},
    {"id":3,"type":"fill","marks":3,
     "text":"What is the output of:\n\nprint(sorted({3,1,4,1,5,9,2,6}))\n\nType exactly as Python would print it.",
     "ah":h("[1, 2, 3, 4, 5, 6, 9]")},
    {"id":4,"type":"fill","marks":3,
     "text":"What is the output of:\n\nprint('hello'[::2])",
     "ah":h("hlo")},
    {"id":5,"type":"fill","marks":3,
     "text":"What does this print?\n\ndef func(a, b=2, *args, **kwargs):\n    print(a, b, args, kwargs)\nfunc(1, 3, 4, 5, x=6)",
     "ah":h("1 3 (4, 5) {'x': 6}")},
    {"id":6,"type":"fill","marks":4,
     "text":"What is the output of:\n\ndef counter():\n    count = 0\n    def inc():\n        nonlocal count\n        count += 1\n        return count\n    return inc\nc = counter()\nprint(c(), c(), c())",
     "ah":h("1 2 3")},
    {"id":7,"type":"code","marks":4,
     "text":"Write a generator function fibonacci(n) that yields first n Fibonacci numbers.\n\nPreloaded: n = 10\n\nPrint the list of first 10 Fibonacci numbers.",
     "preload":"n = 10\n",
     "exp":"[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:\n  matrix = [[1,2,3],[4,5,6],[7,8,9]]\n\nTranspose the matrix using list comprehension.\nPrint the transposed matrix.",
     "preload":"matrix = [[1,2,3],[4,5,6],[7,8,9]]\n",
     "exp":"[[1, 4, 7], [2, 5, 8], [3, 6, 9]]"},
    {"id":9,"type":"code","marks":5,
     "text":"Write a function group_by(lst, key_func) that groups elements by key_func.\n\nPreloaded:\n  words = ['apple','ant','banana','bear','cherry','cat']\n\nGroup by first letter. Print sorted by key.",
     "preload":"words = ['apple','ant','banana','bear','cherry','cat']\n",
     "exp":"{'a': ['apple', 'ant'], 'b': ['banana', 'bear'], 'c': ['cherry', 'cat']}"},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 3 — OOP + Error Handling
# ════════════════════════════════════════════════════════════════
M1_PRACTICE_3 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"What is the output of:\n\nclass A:\n    x = 10\na = A()\na.x = 20\nprint(A.x, a.x)",
     "opts":["10 10","20 20","10 20","20 10"],
     "ah":h("10 20")},
    {"id":2,"type":"mcq","marks":2,
     "text":"What does @staticmethod mean in Python?",
     "opts":[
         "Method can only access class variables",
         "Method does not receive self or cls — belongs to the class namespace",
         "Method is called automatically on class creation",
         "Method cannot be overridden"
     ],
     "ah":h("Method does not receive self or cls — belongs to the class namespace")},
    {"id":3,"type":"fill","marks":3,
     "text":"What is the output of:\n\nsum(i for i in range(1, 101) if i%3==0 or i%5==0)",
     "ah":h("2418")},
    {"id":4,"type":"fill","marks":3,
     "text":"What does this print?\n\ntry:\n    x = int('abc')\nexcept ValueError:\n    print('value error')\nexcept Exception:\n    print('other error')\nfinally:\n    print('done')",
     "ah":h("value error\ndone")},
    {"id":5,"type":"fill","marks":3,
     "text":"What is the output of:\n\nclass MyClass:\n    def __init__(self): self.x = 0\n    def __add__(self, other): return self.x + other.x\na = MyClass(); a.x = 5\nb = MyClass(); b.x = 3\nprint(a + b)",
     "ah":h("8")},
    {"id":6,"type":"fill","marks":4,
     "text":"What is the output of:\n\nclass Animal:\n    def __init__(self, name): self.name = name\nclass Dog(Animal):\n    def __init__(self, name, breed):\n        super().__init__(name)\n        self.breed = breed\nd = Dog('Rex', 'Lab')\nprint(d.name, d.breed)",
     "ah":h("Rex Lab")},
    {"id":7,"type":"code","marks":4,
     "text":"Write a class Stack with methods:\n  push(item), pop(), peek(), is_empty(), size()\n\nRun:\n  s = Stack()\n  s.push(1); s.push(2); s.push(3)\n  print(s.peek())\n  s.pop()\n  print(s.size())",
     "preload":"",
     "exp":"3\n2"},
    {"id":8,"type":"code","marks":4,
     "text":"Write a context manager class Timer using __enter__ and __exit__.\nIt should print 'start' on enter and 'end' on exit.\n\nRun:\n  with Timer():\n      pass",
     "preload":"",
     "exp":"start\nend"},
    {"id":9,"type":"code","marks":5,
     "text":"Implement a LRU Cache class with capacity=2.\nMethods: get(key) returns -1 if not found, put(key, value).\n\nRun:\n  cache = LRUCache(2)\n  cache.put(1, 1)\n  cache.put(2, 2)\n  print(cache.get(1))\n  cache.put(3, 3)\n  print(cache.get(2))",
     "preload":"",
     "exp":"1\n-1"},
]

# ════════════════════════════════════════════════════════════════
# FINAL ASSESSMENT — Tough, primarily coding
# ════════════════════════════════════════════════════════════════
M1_FINAL = [
    {"id":1,"type":"code","marks":3,
     "text":"What is the output of this code?\n\ndef append_item(item, lst=[]):\n    lst.append(item)\n    return lst\nprint(append_item(1))\nprint(append_item(2))\nprint(append_item(3))",
     "preload":"",
     "exp":"[1]\n[1, 2]\n[1, 2, 3]"},
    {"id":2,"type":"code","marks":3,
     "text":"What does this print?\n\ndef make_multiplier(n):\n    return lambda x: x * n\ndouble = make_multiplier(2)\ntriple = make_multiplier(3)\nprint(double(5) + triple(4))",
     "preload":"",
     "exp":"22"},
    {"id":3,"type":"code","marks":4,
     "text":"What is the output of:\n\ndef func(*args, **kwargs):\n    return sum(args) + sum(kwargs.values())\nprint(func(1, 2, 3, a=4, b=5))",
     "preload":"",
     "exp":"15"},
    {"id":4,"type":"mcq","marks":3,
     "text":"Why does this raise a TypeError?\n\nd = {[1,2]: 'list'}",
     "opts":[
         "Lists cannot be dict values",
         "Lists are unhashable so cannot be dict keys",
         "Dict does not support nested types",
         "Syntax error in dict literal"
     ],
     "ah":h("Lists are unhashable so cannot be dict keys")},
    {"id":5,"type":"code","marks":5,
     "text":"What is the complete output of:\n\ndef divide(a, b):\n    try:\n        result = a / b\n    except ZeroDivisionError:\n        print('error')\n        return None\n    else:\n        print('success')\n        return result\n    finally:\n        print('done')\nprint(divide(10, 2))",
     "preload":"",
     "exp":"success\ndone\n5.0"},
    {"id":6,"type":"code","marks":4,
     "text":"Write a generator function gen(n) that yields squares of 0 to n-1.\n\nPrint list(gen(5))",
     "preload":"",
     "exp":"[0, 1, 4, 9, 16]"},
    {"id":7,"type":"code","marks":5,
     "text":"What is the output of:\n\nclass Animal:\n    def __init__(self, name): self.name = name\n    def speak(self): return f'{self.name} speaks'\nclass Dog(Animal):\n    def speak(self): return f'{self.name} barks'\nd = Dog('Rex')\nprint(d.speak())\nprint(isinstance(d, Animal))",
     "preload":"",
     "exp":"Rex barks\nTrue"},
    {"id":8,"type":"code","marks":5,
     "text":"Write a function deep_count(lst, target) that counts occurrences\nof target in a nested list of any depth.\n\nPreloaded:\n  nested = [1, [2, 1], [3, [1, 4]], 1]\n  target = 1\n\nPrint the count.",
     "preload":"nested = [1,[2,1],[3,[1,4]],1]\ntarget = 1\n",
     "exp":"4"},
    {"id":9,"type":"code","marks":5,
     "text":"Implement a decorator validate_positive that raises ValueError\nwith message 'Negative value not allowed' if any argument is negative.\n\nApply it to:\n  def multiply(a, b): return a * b\n\nRun:\n  print(multiply(3, 4))\n  try:\n      multiply(-1, 2)\n  except ValueError as e:\n      print(e)",
     "preload":"",
     "exp":"12\nNegative value not allowed"},
    {"id":10,"type":"code","marks":5,
     "text":"Write a class BankAccount with:\n  - __init__(self, balance)\n  - deposit(amount), withdraw(amount)\n  - withdraw raises ValueError('Insufficient funds') if balance too low\n  - balance property\n\nRun:\n  acc = BankAccount(1000)\n  acc.deposit(500)\n  print(acc.balance)\n  try:\n      acc.withdraw(2000)\n  except ValueError as e:\n      print(e)\n  acc.withdraw(300)\n  print(acc.balance)",
     "preload":"",
     "exp":"1500\nInsufficient funds\n1200"},
    {"id":11,"type":"mcq","marks":3,
     "text":"What is the average time complexity of looking up a key in a Python dictionary?",
     "opts":["O(n)","O(log n)","O(1)","O(n²)"],
     "ah":h("O(1)")},
    {"id":12,"type":"mcq","marks":3,
     "text":"What will this print?\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
     "opts":["[1, 2, 3]","[1, 2, 3, 4]","[4]","Error"],
     "ah":h("[1, 2, 3, 4]")},
]

# ── MAPS ─────────────────────────────────────────────────────────
M1_PRACTICE_TESTS = {1: M1_PRACTICE_1, 2: M1_PRACTICE_2, 3: M1_PRACTICE_3}
M1_FINAL_TEST     = {1: M1_FINAL}