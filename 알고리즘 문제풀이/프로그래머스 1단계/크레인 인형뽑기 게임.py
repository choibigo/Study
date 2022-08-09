def solution(board, moves):
    
    # 4 3 1 1 3 2 0 4
    
    stack_list = [-999]
    count = 0
    
    for col in moves:
        for row in range(1, len(board)+1):
            num = board[row-1][col-1]
            if num != 0:
                board[row-1][col-1] = 0
                if stack_list[-1] == num:
                    stack_list.pop()
                    count +=2
                else:
                    stack_list.append(num)
                
                break
    
    return count