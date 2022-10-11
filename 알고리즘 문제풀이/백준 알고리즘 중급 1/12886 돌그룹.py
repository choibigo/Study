import input_setting

from collections import deque
import sys

a,b,c = map(int, input().split())
total_stone = sum([a,b,c])
check = [[False for _ in range(total_stone+1)] for _ in range(total_stone+1)]

nodes = deque()
nodes.append([a,b])
check[a][b] = False

while nodes:
    
    x,y = nodes.popleft()
    z = total_stone - (x+y)

    if x==y==z:
        print(1)
        sys.exit()


    for a, b in [(x,y),(x,z),(y,z)]:
        if a<b:
            b = b-a
            a = a*2
        elif a>b:
            a = a-b
            b = b*2
        else:
            continue

        c = total_stone - (a+b)

        x = min(a, b, c)
        y = max(a, b, c)

        if not check[x][y]:
            check[x][y] = True
            nodes.append([x,y])

print(0)