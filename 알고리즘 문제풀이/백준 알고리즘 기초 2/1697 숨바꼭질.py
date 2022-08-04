import input_setting

from collections import deque

start, end = map(int, input().split())

max_count = 100001
res = [-1] * (max_count+1)

res[start] = 0
nodes = deque()
nodes.append(start)

while nodes:
    pos = nodes.popleft()

    if pos == end:
        print(res[pos])
        break

    for next in [pos+1, pos*2, pos-1]:
        if 0<=next<max_count:
            if res[next] == -1:
                res[next] = res[pos] + 1
                nodes.append(next)

            else:
                if res[next] > res[pos] + 1:
                    res[next] = res[pos] + 1
                    nodes.append(next)
