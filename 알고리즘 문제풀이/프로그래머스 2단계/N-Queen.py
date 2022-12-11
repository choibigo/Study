def solution(n):

    board = [0 for _ in range(n)]
    
    def check(x, board):
        for i in range(x):
            if board[i] == board[x] or abs(x-i) == abs(board[x]-board[i]):
                return False
        return True
    
    result = 0
    def DFS(v, board):
        nonlocal result
        
        if v == n:
            result +=1
            return 
        
        for i in range(n):
            board[v] = i
            if check(v, board):
                DFS(v+1, board)
                
    DFS(0, board)
    return result