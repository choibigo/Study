def solution(land):
    
    
    for row in range(1, len(land)):
        for col in range(len(land[0])):
            land[row][col] = max(land[row-1][:col]+land[row-1][col+1:]) + land[row][col]
    
    return max(land[-1])