def solution(s):
    size = len(s)
    answer = 0
    
    for left in range(size):
        for right in range(size-1, -1, -1):
            
            if right < left:
                break
            
            if s[left] != s[right]:
                continue
            
            if s[left:right+1] == s[left:right+1][::-1]:
                if answer < right-left:
                    answer = right-left
                
                

    return answer +1