def solution(n):
    
    n = list(str(n))
    n.sort(reverse = True)
    n = "".join(n)
    
    return int(n)