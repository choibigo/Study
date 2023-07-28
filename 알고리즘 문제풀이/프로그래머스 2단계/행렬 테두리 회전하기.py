def solution(rows, cols, queries):
    
    board = [[r*cols+c for c in range(1,cols+1)] for r in range(rows)]
    answer = list()
    
    for querie in queries:
        left_y, left_x, right_y, right_x = querie
        
        left_y-=1
        left_x-=1
        right_y-=1
        right_x-=1
        
        temp = board[left_y][left_x]
        min_value = temp
        
        # Left Line
        for y in range(left_y, right_y):
            board[y][left_x] = board[y+1][left_x]
            min_value = min(board[y][left_x], min_value)
            
        # Bottom Line
        for x in range(left_x, right_x):
            board[right_y][x] = board[right_y][x+1]
            min_value = min(board[right_y][x], min_value)
        
        # Right Line
        for y in range(right_y, left_y, -1):
            board[y][right_x] = board[y-1][right_x]
            min_value = min(board[y][right_x], min_value)
        
        # Top Line
        for x in range(right_x, left_x, -1):
            board[left_y][x] = board[left_y][x-1]
            min_value = min(board[left_y][x], min_value)
        
        board[left_y][left_x+1] = temp
        
        answer.append(min_value)
        
    # print(*board, sep="\n")
        
    return answer