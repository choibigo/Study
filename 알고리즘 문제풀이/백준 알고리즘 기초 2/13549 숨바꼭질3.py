import input_setting

from collections import deque

start, end = map(int, input().split())

max_size = 100000
res = [-1] * (max_size + 1)
res[start] = 0

nodes = deque()
nodes.append(start)

while nodes:
    pos = nodes.popleft()

    if pos == end:
        print(res[pos])
        break

    for next, time in [[pos + 1, 1],[pos * 2, 0],[pos - 1, 1]]:
        if 0<=next<=max_size:
            if res[next] == -1:
                res[next] = res[pos] + time
                nodes.append(next)
            else:
                if res[next] > res[pos] + time:
                    res[next] = res[pos] + time
                    nodes.append(next)