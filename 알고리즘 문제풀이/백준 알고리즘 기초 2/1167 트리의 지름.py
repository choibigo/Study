import input_setting

from collections import defaultdict
import sys

def DFS(v, weight):
    
    for n_w in graph[v]:
        next, nweight  = n_w

        if visited[next] == -1:
            visited[next] = 0
            res[next] = weight + nweight
            DFS(next, weight + nweight)

            

n = int(input())
graph = defaultdict(list)
parent_node = [-1] * (n+1)

for _ in range(n):
    temp = list(map(int, sys.stdin.readline().split()))

    parent = temp[0]
    
    ttt = list()
    for i in range(1, len(temp)-1):
        ttt.append(temp[i])
        if i % 2 == 0:
            graph[parent].append(ttt)
            ttt= list()


res = [-1] * (n+1)
res[1] = 0
visited = [-1] * (n+1)
visited[1] = 0

DFS(1,0)

n1 = -1
n1_weight = -1

for i in range(n+1):
    if res[i] > n1_weight:
        n1 = i
        n1_weight = res[i]


res = [-1] * (n+1)
res[n1] = 0
visited = [-1] * (n+1)
visited[n1] = 0

DFS(n1,0)

n2 = -1
n2_weight = -1

for i in range(n+1):
    if res[i] > n2_weight:
        n2 = i
        n2_weight = res[i]

print(n2_weight)
