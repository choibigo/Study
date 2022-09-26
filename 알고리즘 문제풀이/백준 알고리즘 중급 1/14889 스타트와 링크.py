import input_setting

n = int(input())
board = [list(map(int, input().split())) for _ in range(n)]
check = [False]*n

min_val = float("inf")

def DFS(index, s, res):

    global min_val

    if index == n//2:
        start = 0
        link = 0

        for row in range(n):
            for col in range(n):
                if row in res and col in res:
                    start += board[row][col]
                if row not in res and col not in res:
                    link += board[row][col]

        diff = abs(start-link)
        if min_val > diff:
            min_val = diff
        return 

    for i in range(s, n):
        if check[i]==False:
            DFS(index+1, i+1, res+[i])
    

res = list()
DFS(0, 0, res)

print(min_val)
