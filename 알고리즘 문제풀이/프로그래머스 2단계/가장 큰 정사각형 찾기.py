def solution(board):
    
    rows = len(board)
    cols = len(board[0])
    answer = 0
    
    if len(board) == 1 and len(board[0]) == 1:
        if board[0][0] == 0:
            return 0
        elif board[0][0] == 1:
            return 1
        
    for row in range(1, rows):
        for col in range(1, cols):
            if board[row][col] != 0:
                board[row][col] = min(board[row-1][col], board[row][col-1], board[row-1][col-1]) + 1
            answer = max(answer, board[row][col])
        
    return answer**2