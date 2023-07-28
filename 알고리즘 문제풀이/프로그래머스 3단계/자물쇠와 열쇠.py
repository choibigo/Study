def attach(x, y, M, key, board):
    for i in range(M):
        for j in range(M):
            board[x+i][y+j] += key[i][j]

def detach(x, y, M, key, board):
    for i in range(M):
        for j in range(M):
            board[x+i][y+j] -= key[i][j]

def check(board, M, N):
    for i in range(N):
        for j in range(N):
            if board[M+i][M+j] != 1:
                return False;
    return True
    
def solution(key, lock):

    m = len(key)
    n = len(lock)
    lock_padding = [[0 for _ in range(2*m+n)] for _ in range(2*m+n)]
    
    for row in range(n):
        for col in range(n):
            lock_padding[m+row][m+col] = lock[row][col]
    
    for _ in range(4):
        key = [l for l in zip(*key[::-1])]
        for row in range(1, n+m):
            for col in range(1, n+m):
                attach(col, row, m, key, lock_padding)
                if check(lock_padding, m, n):
                    return True
                detach(col, row, m, key, lock_padding)
    
    return False