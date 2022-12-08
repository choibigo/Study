import input_setting





import sys
sys.setrecursionlimit(10 ** 6)
for _ in range(int(sys.stdin.readline().strip())):
    n = int(sys.stdin.readline().strip())
    graph = [0]+list(map(int, sys.stdin.readline().split()))
    visited = [False for _ in range(n+1)]
    
    def DFS(v):
        global result
        visited[v] = True
        cycle.append(v)
        nextv = graph[v]

        if visited[nextv]:
            if nextv in cycle:
                result += cycle[cycle.index(nextv):]
            return 
        else:
            DFS(nextv)

    result = list()
    for i in range(1, n+1):
        if not visited[i]:
            cycle = []
            DFS(i)

    print(n - len(result))