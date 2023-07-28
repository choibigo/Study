def solution(numbers, target):
    count = 0
    
    def DFS(v, cal_val):
        nonlocal count
        
        if v == len(numbers):
            if cal_val == target:
                count +=1
            return 
        
        DFS(v+1, cal_val+numbers[v])
        DFS(v+1, cal_val-numbers[v])
    
    DFS(0, 0)
    return count
    