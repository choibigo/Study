def solution(n):

    answer = 0
    
    for i in range(1, n+1):
        sum_val = 0
        for j in range(i, n+1):
            sum_val += j
            
            if sum_val == n:
                answer+=1
                break
            elif sum_val > n:
                break
                
    return answer
                