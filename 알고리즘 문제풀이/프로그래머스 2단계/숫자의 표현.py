def solution(n):
    
    answer = 0
    
    for i in range(1, n+1):
        sum_value = 0
        for j in range(i, n+1):
            sum_value += j

            if sum_value == n:
                answer +=1
                break
            elif sum_value > n:
                break
                
    return answer