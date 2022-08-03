import input_setting

import sys
sys.setrecursionlimit(10 ** 6)
input = sys.stdin.readline

def DFS(v, color):
    
    global flag

    if flag == True:
        return
    
    res[v] = color

    for g in graph[v]:
        if res[g] == 0:
            DFS(g, -color)
        else:
            if res[g] == color:
                flag = True
                return

case = int(input())

for _ in range(case):
    node_count, edge_count = map(int, input().split())

    graph = [[] for _ in range(node_count+1)]
    res = [0] * (node_count + 1)

    for _ in range(edge_count):
        a, b = map(int ,input().split())
        graph[a].append(b)
        graph[b].append(a)

    flag = False

    for i in range(1, node_count+1):
        if res[i] == 0:
            DFS(i, 1)

            if flag == True:
                break

    if flag == False:
        print("YES")
    else:
        print("NO")
