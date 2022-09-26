from collections import defaultdict


def solution(tickets):
    visited = defaultdict(list)
    graph = defaultdict(list)
    start = "ICN"
    goal = len(tickets) + 1
    answer = []

    for ticket in tickets:
        graph[ticket[0]].append(ticket[1])
        visited[ticket[0]].append(False)

    def dfs(now, path):
        nonlocal goal

        if len(path) == goal:
            answer.append(path)
            return 0
        for j in range(len(graph[now])):
            if not visited[now][j]:
                nxt = graph[now][j]
                visited[now][j] = True
                dfs(nxt, path + [nxt])
                visited[now][j] = False

    for i in range(len(graph[start])):
        airport = graph[start][i]
        visited[start][i] = True
        dfs(airport, [start, airport])
        visited[start][i] = False

    answer.sort()
    return answer[0]

