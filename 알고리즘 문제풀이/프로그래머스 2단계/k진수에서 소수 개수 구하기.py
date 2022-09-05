def k_num_trase(num, k):
    if num < k :
        return str(num)
    
    return k_num_trase(num//k, k) + str(num%k)

def prime(num):
    
    if num == 1:
        return False
    
    for i in range(2, int(num**(1/2))+1):
        if num%i == 0:
            return False
        
    return True

def solution(n, k):
    
    k_num = k_num_trase(n, k)
    temp = k_num.split("0")
    
    answer = 0
    for num in temp:
        if len(num) != 0:
            if prime(int(num)):
                answer +=1
    
    return answer