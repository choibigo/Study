def solution(arr1, arr2):
    
    answer = list()
    
    for arr1_row in arr1:
        temp = list()
        for arr2_row in zip(*arr2):
            value = 0
            for a, b in zip(arr1_row, arr2_row):
                value += (a*b)
            temp.append(value)
            
        answer.append(temp)
    
    return answer