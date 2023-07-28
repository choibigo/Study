def k_num(num, k):
    if num<k:
        return str(num)
    
    return k_num(num//k, k) + str(num%k)

def is_prime(num):
    if num==1:
        return False
    
    for i in range(2, int(num**0.5)+1):
        if num%i == 0:
            return False
    
    return True

def solution(num, k):

    return sum([1 for n in k_num(num, k).split("0") if len(n) and is_prime(int(n))]) 
    