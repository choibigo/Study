def solution(gems):
    
    gems_len = len(gems)
    gems_set_len = len(set(gems))
    
    answer = [0, gems_len-1]
    
    left = 0
    right = 0
    
    info = {gems[0] : 1}
    
    while left<=right:
        if len(info) < gems_set_len:
            right += 1
            if right == gems_len:
                break
            
            if gems[right] in info:
                info[gems[right]] += 1
            else:
                info[gems[right]] = 1
            
        else:
            if (right - left) < (answer[1] - answer[0]):
                answer[0] = left
                answer[1] = right
            
            if info[gems[left]] == 1:
                del info[gems[left]]
            else:
                info[gems[left]] -= 1
            
            left += 1
            
    return [answer[0]+1, answer[1]+1]