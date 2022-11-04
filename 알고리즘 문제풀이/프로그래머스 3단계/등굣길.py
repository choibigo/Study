def solution(m, n, puddles):

    board = [[0 for _ in range(m)] for _ in range(n)]
    
    for x,y in puddles:
        board[y-1][x-1] = -1
    
    for col in range(1, m):
        if board[0][col] == -1:
            break
        board[0][col] = 1
        
    for row in range(0, n):
        if board[row][0] == -1:
            break
        board[row][0] = 1
    
    for row in range(1, n):
        for col in range(1, m):
            if board[row][col] != -1:
                top = board[row-1][col]
                left = board[row][col-1]

                if top == -1 and left == -1:
                    board[row][col] = 0
                elif top == -1 and left != -1:
                    board[row][col] = left % 1000000007
                elif top != -1 and left == -1:
                    board[row][col] = top % 1000000007
                else:
                    board[row][col] = (top + left) % 1000000007
    
    return board[-1][-1]