import input_setting


# Need To Change
import sys
from collections import deque
for _ in range(int(input())):
    nodes, paths = map(int, input().split())
    cost = [0] + list(map(int,input().split()))
    cntLink = [-1] + [0]*(nodes)

    graph = [[] for _ in range(nodes+1)]
    for _ in range(paths):
        a, b = map(int, sys.stdin.readline().split())
        graph[a].append(b)
        cntLink[b] +=1
    end = int(input())

    dp = [0 for _ in range(nodes+1)]
    q = deque()
    for i in range(1, nodes+1):
        if cntLink[i] == 0:
            q.append(i)
            dp[i] = cost[i]

    while q:
        current_v = q.popleft()

        for next_v in graph[current_v]:
            cntLink[next_v] -=1
            dp[next_v] = max(dp[next_v], dp[current_v]+cost[next_v])
            if cntLink[next_v] == 0:
                q.append(next_v)

        if cntLink[end]==0:
            print(dp[end])
            break