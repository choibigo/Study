import input_setting

import sys

n, s = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))

count = 0

def DFS(index, res, n, s, num_list):
    global count

    if index == n:
        if sum(res) == s and len(res) > 0:
            count +=1
        return 

    DFS(index+1, res+[num_list[index]], n, s, num_list)
    DFS(index+1, res, n, s, num_list)

DFS(0, list(), n, s, num_list)

print(count)