import input_setting

import sys

case = int(input())

def func(num1, num2):
    x = num1
    y = num2
    while(y):
        x, y = y, x%y
    
    result = (num1 * num2) / x

    return result

for _ in range(case):
    m, n, x, y = map(int, input().split())
    temp = m*n

    while x <= temp:
        if x%n == y%n:
            print(x)
            break

        x+=m

    if x > temp:
        print(-1)