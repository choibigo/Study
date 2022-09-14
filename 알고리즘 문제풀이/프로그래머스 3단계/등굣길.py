def solution(m, n, puddles):
    
    res = [[0 for _ in range(m) ] for _ in range(n)]
    
    num = 1000000007
    
    for puddle in puddles:
        res[puddle[1]-1][puddle[0]-1] = -1
    
    
    temp = 1
    for col in range(1, m):
        if res[0][col] == -1:
            temp = -1
        res[0][col] = temp
    
    temp = 1
    for row in range(1, n):
        if res[row][0] == -1:
            temp = -1
        res[row][0] = temp
        
    for row in range(1, n):
        for col in range(1, m):
            if res[row][col] == -1:
                continue
            else:
                left = res[row][col-1]
                up = res[row-1][col]
                
                if left == -1 and up == -1:
                    res[row][col] = 0
                elif left == -1:
                    res[row][col] += (up%num)
                elif up == -1:
                    res[row][col] += (left%num)
                else:
                    res[row][col] +=((up+left)%num)

    return res[-1][-1]