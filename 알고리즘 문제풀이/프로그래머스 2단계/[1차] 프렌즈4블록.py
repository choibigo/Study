def block_check(rows, cols, board):
    block_points = list()
    
    for row in range(rows-1):
        for col in range(cols-1):
            
            my_block = board[row][col]
    
            if my_block == "#":
                continue
            
            if board[row][col] == board[row+1][col] == board[row][col+1] == board[row+1][col+1]:
                block_points.append((col, row))
                
    
    for block_point in block_points:
        col, row = block_point
        board[row][col] = "#"
        board[row+1][col] = "#"
        board[row][col+1] = "#"
        board[row+1][col+1] = "#"
    
    
    return block_points, board

def block_move(rows, cols, board, block_points):
    
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == "#":
                temp = ["#"] + board[row][:col]
                board[row][:col+1] = temp
    
    return board

def solution(rows, cols, board):
    
    board = [list(temp) for temp in zip(*board)]
    
    while True:
        block_points, board = block_check(cols, rows, board)
        
        if len(block_points) == 0:
            break
        
        board = block_move(cols, rows, board, block_points)
    
    
    return sum([board_row.count("#") for board_row in board])
