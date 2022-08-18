def solution(n):
    
    temp = "수박"
    count = (n+1) / 2
    anwer = temp * int(count)
    
    return anwer[:n]