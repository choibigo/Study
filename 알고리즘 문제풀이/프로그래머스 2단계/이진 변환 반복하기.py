def two_num(n):
    if n <2:
        return str(n)
    
    return two_num(n//2)+str(n%2)

def solution(target):
    
    answer = [0,0]
    zero_count = target.count("0")

    while target != "1":
        zero_count = target.count("0")
        answer[1] += zero_count
        
        target = two_num(len(target)-zero_count)
        answer[0] += 1
    
    return answer