def solution(word):

    
    alpha = ['A', 'E', 'I', 'O', 'U']
    total_list = list()
    res = list()
    
    def DFS(v):
        
        if v <= len(alpha):
            total_list.append("".join(res))

            if v == len(alpha):
                return
            
        for a in alpha:
            res.append(a)
            DFS(v+1)
            res.pop()
    
    DFS(0)
    total_list.sort()
    
    return total_list.index(word)
    