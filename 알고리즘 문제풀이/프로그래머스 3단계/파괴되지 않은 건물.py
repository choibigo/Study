def solution(board, skill):

    rows = len(board)
    cols = len(board[0])
    
    temp = [[0 for _ in range(cols+1)] for _ in range(rows+1)]
    
    for t, r1, c1, r2, c2, value in skill:
        temp[r1][c1]+= value if t == 2 else -value
        temp[r1][c2+1]+= -value if t == 2 else value
        temp[r2+1][c1]+= -value if t == 2 else value
        temp[r2+1][c2+1]+= value if t==2 else -value
        
    for row in range(rows):
        for col in range(cols):
            temp[row][col+1] += temp[row][col] 
    
    for row in range(rows):
        for col in range(cols):
            temp[row+1][col] += temp[row][col]
    
    answer = 0
    for row in range(rows):
        for col in range(cols):
            if board[row][col] + temp[row][col] > 0:
                answer +=1
    return answer