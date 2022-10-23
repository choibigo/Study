def solution(arr1, arr2):
    
    answer = [[0 for _ in range(len(arr2[0]))] for _ in range(len(arr1))]
    for r, arr1_row in enumerate(arr1):
        for c, arr2_col in enumerate(zip(*arr2)):
            answer[r][c] = sum([a*b for a,b in zip(arr1_row, arr2_col)])
    
    return answer
            