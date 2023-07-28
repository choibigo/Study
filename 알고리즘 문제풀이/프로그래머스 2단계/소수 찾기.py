def is_prime(num):
    if num<=1:
        return False
    for i in range(2, int(num**0.5)+1):
        if num%i==0:
            return False
    return True

def solution(numbers):
    
    res = [False] * len(numbers)
    answer = set()    
    def DFS(n):
        if len(n) and is_prime(int(n)):
            answer.add(int(n))
        
        for i in range(len(numbers)):
            if not res[i]:
                res[i] = True
                DFS(n+numbers[i])
                res[i] = False
    
    DFS('')
    return len(answer)