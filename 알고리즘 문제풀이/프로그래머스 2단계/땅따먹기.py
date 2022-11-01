def solution(land):
    
    res = [[0 for _ in range(len(land[0]))] for _ in range(len(land))]
    res[0] = land[0][:]
    
    for row in range(1, len(land)):
        for col in range(len(land[0])):
            temp = res[row-1][:]
            temp[col] = -1
            res[row][col] = max(temp)+land[row][col]
    
    
    return max(res[-1])