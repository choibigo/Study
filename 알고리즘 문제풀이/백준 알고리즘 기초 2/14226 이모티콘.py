import input_setting

from collections import deque

n = int(input())
res = [[-1 for _ in range(n+1)] for _ in range(n+1)]

nodes = deque()
nodes.append([1,0])
res[1][0] = 0

while nodes:
    count, clip = nodes.popleft()

    if res[count][count] == -1:
        res[count][count] = res[count][clip] + 1
        nodes.append([count, count])

    if count+clip <= n and res[count + clip][clip] == -1:
        res[count+clip][clip] = res[count][clip] + 1
        nodes.append([count+clip, clip])

    if count-1 >= 0 and res[count-1][clip] == -1 :
        res[count-1][clip] = res[count][clip] + 1
        nodes.append([count-1, clip])


import sys
result = sys.maxsize
for r in res[n]:
    if r != -1:
        if result > r:
            result = r

print(result)