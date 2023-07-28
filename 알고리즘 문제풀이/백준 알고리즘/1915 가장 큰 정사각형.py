import input_setting

rows, cols = list(map(int, input().split()))

board = [list(map(int, list(input()))) for _ in range(rows)]
res = board[:]

for row in range(1, rows):
    for col in range(1, cols):

        if board[row][col] == 1:

            temp = [res[row-1][col], res[row][col-1], res[row-1][col-1]]
            res[row][col] = min(temp) + 1

max_val = max(list(map(max, res)))

print(max_val**2)