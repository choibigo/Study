import input_setting

for _ in range(int(input())):
    rows = int(input())
    cols = int(input())

    board = [[c+1 for c in range(cols)] for _ in range(rows+1)]
    for row in range(1, rows+1):
        for col in range(1, cols):
            board[row][col] = board[row-1][col] + board[row][col-1]
    print(board[-1][-1])