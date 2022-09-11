def solution(n):
    answer = ""
    res = ["1", "2", "4"]
    
    while n >0:
        n = n-1
        answer = res[n%3] + answer
        n //= 3
    
    return answer