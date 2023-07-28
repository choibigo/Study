def solution(citations):
    
    citations.sort(reverse=True)

    for i, num in enumerate(citations):
        
        if i >= num:
            return i
        
    return len(citations)
        
