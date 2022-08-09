def solution(absolutes, signs):
    
    answer = 0
    
    for i in range(len(absolutes)):
        temp = absolutes[i]
        if signs[i] == False:
            temp = -temp
            
        answer += temp
        
    return answer