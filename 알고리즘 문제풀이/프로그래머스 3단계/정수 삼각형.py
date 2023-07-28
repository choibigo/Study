def solution(triangle):
    
    answer = [[0 for _ in range(i+1)] for i in range(len(triangle))]
    answer[0][0] = triangle[0][0]
    
    for row in range(1, len(triangle)):
        for col in range(row+1):
            if col == 0:
                answer[row][col] = answer[row-1][col]+triangle[row][col]
            elif col == row:
                answer[row][col] = answer[row-1][col-1]+triangle[row][col]
            else:
                answer[row][col] = max(answer[row-1][col-1], answer[row-1][col]) + triangle[row][col]
    
    return max(answer[-1])
            