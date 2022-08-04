import input_setting

from collections import deque
import copy

start, end = map(int, input().split())

max_size = 100000
res = [-1] * (max_size + 1)
res[start] = 0
path_res = [-1] * (max_size + 1)

nodes = deque()
nodes.append(start)

while nodes:
    pos = nodes.popleft()

    if pos == end:
        print(res[pos])
        break

    for next in [pos+1, pos*2, pos-1]:
        if 0<=next<=max_size:
            if res[next] == -1:
                res[next] = res[pos] + 1
                nodes.append(next)
                path_res[next] = pos

            else:
                if res[next] > res[pos] + 1:
                    res[next] = res[pos] + 1
                    nodes.append(next)
                    path_res[next] = pos

temp_list = list()
index = end
while True:
    if path_res[index] == -1:
        break

    else:
        index = path_res[index]
        temp_list.append(index)

print(*temp_list[::-1], end)