import input_setting

import sys

def DFS(index, s, res, target_list):
    if index == 6:
        print(*res, sep=" ")
        return 

    for i in range(s, k):
        DFS(index+1, i+1, res+[target_list[i]], target_list)

while(True):
    target_list = list(map(int, sys.stdin.readline().split()))

    if len(target_list) < 2:
        break

    k = target_list.pop(0)

    res=list()
    DFS(0, 0, res, target_list)
    print()


    