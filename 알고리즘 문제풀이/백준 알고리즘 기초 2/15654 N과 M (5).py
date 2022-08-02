import input_setting

import sys

def DFS():
    if len(res) == count:
        print(*res)
        return
    
    for i in range(num_count):
        if ch[i] == 0:
            ch[i] = 1
            res.append(num_list[i])
            DFS()
            ch[i] = 0
            res.pop()


num_count, count = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))
num_list.sort()

res = list()
ch = [0] * (num_count)

DFS()
