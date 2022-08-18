def solution(s):
    
    if len(s) == 6 or len(s) == 4:
        for t in s:
            if t.isalpha():
                return False
    else:
        return False
            
    return True
    
