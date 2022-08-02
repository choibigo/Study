import input_setting

import sys

def DFS(s):
    if len(res) == count:
        print(*res)
        return
    
    for i in range(s, num_count):
        res.append(num_list[i])
        DFS(i+1)
        res.pop()

num_count, count = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))
num_list.sort()

res = list()
ch = [0] * (num_count)

DFS(0)
