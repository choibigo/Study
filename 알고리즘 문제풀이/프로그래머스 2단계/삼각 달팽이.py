def solution(n):
    res = [[0 for _ in range(i)] for i in range(1, n+1)]
    
    pos = (0,-1)
    num = 1
    
    temp_n = n
    for i in range(n):
        for j in range(temp_n):
            if i%3 == 0:
                nx = pos[0]
                ny = pos[1]+1
            elif i%3 == 1:
                nx = pos[0]+1
                ny = pos[1]
            elif i%3 == 2:
                nx = pos[0]-1
                ny = pos[1]-1
            
            res[ny][nx] = num
            num+=1
            pos = (nx, ny)
            
        temp_n -=1
    
    answer = list()
    for res_row in res:
        answer.extend(res_row)
    
    
    return answer