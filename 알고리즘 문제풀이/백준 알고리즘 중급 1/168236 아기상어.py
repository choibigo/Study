from collections import deque

n = int(input())
board = [list(map(int, input().split())) for _ in range(n)]
move_list = [(0, -1),(1,0),(0,1),(-1,0)]

INF = 1e+9
shark_size = 2
now_x = 0
noy_y = 0

for row in range(n):
    for col in range(n):
        if board[row][col] == 9:
            now_x, now_y = col, row
            board[row][col] = 0

def BFS():
    queue = deque([(now_x, now_y)])
    visited = [[-1]*n for _ in range(n)]
    visited[now_y][now_x] = 0
    while queue:
        x, y = queue.popleft()

        for move in move_list:
            nx = move[0] + x
            ny = move[1] + y

            if 0 <= nx < n and 0 <= ny < n:
                if shark_size >= board[ny][nx] and visited[ny][nx] == -1:
                    visited[ny][nx] = visited[y][x] + 1
                    queue.append((nx, ny))
    return visited

def solve(visited):
    x = 0
    y = 0
    min_distance = INF

    for row in range(n):
        for col in range(n):
            if visited[row][col] != -1 and 1<=board[row][col]<shark_size:
                if visited[row][col] < min_distance:
                    min_distance = visited[row][col]
                    x = col
                    y = row

    if min_distance == INF:
        return False

    else:
        return x, y, min_distance


answer = 0
food = 0

while True:
    result = solve(BFS())

    if not result:
        print(answer)
        break

    now_x, now_y = result[0], result[1]
    answer += result[2]
    board[now_y][now_x] = 0
    food += 1

    if food >= shark_size:
        food = 0
        shark_size +=1