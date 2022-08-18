def solution(s):
    
    s = list(s.lower())
    
    index = 0
    for i in range(len(s)):
        if s[i] == " ":
            index = 0
        
        else:
            if index % 2 == 0:
                s[i] = s[i].upper()
            index +=1
    
    
    return "".join(s)