def solution(s):
    
    s = list(map(int, s.split(" ")))
    return f"{min(s)} {max(s)}"