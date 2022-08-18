def solution(arr1, arr2):
    
    
#     cols = len(arr1[0])
#     rows = len(arr1)
    
#     answer = [[0 for _ in range(cols)] for _ in range(rows)]
    
#     for row in range(rows):
#         for col in range(cols):
#             answer[row][col] = arr1[row][col] + arr2[row][col]
    # return answer
    
    return [[ c+d for c,d in zip(a, b)] for a,b in zip(arr1, arr2)]