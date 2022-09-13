def solution(triangle):
    
    for row in range(1, len(triangle)):
        for col in range(len(triangle[row])):
            if col == 0:
                triangle[row][col] = triangle[row-1][0] + triangle[row][0]
            elif col == len(triangle[row])-1:
                triangle[row][col] = triangle[row-1][-1] + triangle[row][-1]
            else:
                triangle[row][col] = max(triangle[row-1][col-1], triangle[row-1][col]) + triangle[row][col]
    
    return max(triangle[-1])