def solution(n):
    battery = 0
    while n:
        battery += n%2
        n //=2
    
    return battery