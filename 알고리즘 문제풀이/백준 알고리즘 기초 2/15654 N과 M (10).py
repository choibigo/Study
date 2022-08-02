import input_setting

import sys

def DFS(s):
    if len(res) == count:
        print(*res)
        return

    overlap_num = 0
    for i in range(s, num_count):
        if num_list[i] != overlap_num:
            overlap_num = num_list[i]
            res.append(num_list[i])
            DFS(i+1)
            res.pop()

num_count, count = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))
num_list.sort()

res = list()
DFS(0)
