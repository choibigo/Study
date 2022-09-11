def place_check(board):
    
    check_list1 = [(0,-1),(1,0),(0,1),(-1,0)]
    check_list2 = [(1,-1),(1,1),(-1,1),(-1,-1)]
    check_list3 = [(0,-2),(2,0),(0,2),(-2,0)]
    
    for row in range(5):
        for col in range(5):
            if board[row][col] == "P":
                for check1 in check_list1:
                    nx = col + check1[0]
                    ny = row + check1[1]

                    if 0<=nx<=4 and 0<=ny<=4:
                        if board[ny][nx] == "P":
                            return 0
                        
                for check2 in check_list2:
                    nx = col + check2[0]
                    ny = row + check2[1]
                
                    if 0<=nx<=4 and 0<=ny<=4:
                        if board[ny][nx] == "P":
                            if check2 == (1,-1):
                                if board[ny+1][nx] != "X" or board[ny][nx-1] != "X":
                                    return 0
                            elif check2 == (1,1):
                                if board[ny-1][nx] != "X" or board[ny][nx-1] != "X":
                                    return 0
                            elif check2 == (-1,1):
                                if board[ny-1][nx] != "X" or board[ny][nx+1] != "X":
                                    return 0
                            elif check2 == (-1,-1):
                                if board[ny+1][nx] != "X" or board[ny][nx+1] != "X":
                                    return 0
                            
                
                for check3 in check_list3:
                    nx = col + check3[0]
                    ny = row + check3[1]
                    
                    if 0<=nx<=4 and 0<=ny<=4:
                        if board[ny][nx] == "P":
                            # print(f"{col},{row} <=> {nx},{ny}")
                            if check3 == (0,-2):
                                if board[ny+1][nx] != "X":
                                    return 0
                            elif check3 == (2,0):
                                if board[ny][nx-1] != "X":
                                    return 0
                            elif check3 == (0,2):
                                if board[ny-1][nx] != "X":
                                    return 0
                            elif check3 == (-2,0):
                                if board[ny][nx+1] != "X":
                                    return 0
    return 1
    
    
def solution(places):
    answer = list()
    
    
    for place in places:
        answer.append(place_check(place))
    
    # print(answer)
    
    return answer