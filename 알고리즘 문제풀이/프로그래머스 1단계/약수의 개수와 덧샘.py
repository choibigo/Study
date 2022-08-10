def div_count(num):
    count = 0
    for i in range(1, num+1):
        if num % i == 0:
            count +=1
    
    return count

def solution(left, right):
    answer = 0
    
    for num in range(left, right+1):
        
        temp = div_count(num)
        if temp % 2 == 0:
            answer += num
        elif temp % 2 == 1:
            answer -= num
    
    return answer