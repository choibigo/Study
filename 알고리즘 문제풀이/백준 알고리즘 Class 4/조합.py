import input_setting

from math import factorial
n, m = map(int, input().split())
up = factorial(n)
down = (factorial(n - m)) * (factorial(m))
print(up // down)