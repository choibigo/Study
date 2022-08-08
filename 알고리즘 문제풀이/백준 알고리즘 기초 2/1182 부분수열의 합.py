import input_setting

import sys

def DFS(v):

    global count

    if v == n:
        if len(res) != 0:
            if sum(res) == s:
                count+=1
        return 

    res.append(num_list[v])
    DFS(v+1)
    res.pop()

    DFS(v+1)

n, s = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))

count = 0
res = list()
DFS(0)
print(count)