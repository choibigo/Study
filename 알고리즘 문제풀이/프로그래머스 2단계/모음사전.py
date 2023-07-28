def solution(target_word):
    
    alpha = ['A', 'E', 'I', 'O', 'U']
    
    count = 0
    answer = 0
    def DFS(word):
        nonlocal count,answer
        if word == target_word:
            answer = count
            return        
        count +=1
        if len(word) == 5:
            return
        
        for a in alpha:
            DFS(word+a)
            
    DFS("")
    return answer