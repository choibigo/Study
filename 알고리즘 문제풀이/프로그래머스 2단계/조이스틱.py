def solution(name):
    
    info = {chr(i):i-65 for i in range(65, 91)}

    answer = 0
    min_move = len(name)
    for i, c in enumerate(name):
        answer += min(26-info[c], info[c])
        
        ni = i+1
        while ni<len(name) and name[ni] == 'A':
            ni += 1
            
        min_move = min(min_move, i+i + len(name)-ni, i + 2*(len(name)-ni))
        
    answer += min_move
    
    return answer