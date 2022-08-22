def DFS(v):
    global count
    
    if v == number_size:
        if sum(res) == g_target:
            count +=1
        return 
    
    res.append(g_numbers[v])
    DFS(v+1)
    res.pop()
    
    res.append(g_numbers[v] * -1)
    DFS(v+1)
    res.pop()
    
    

res = list()
count = 0
number_size = 0
g_numbers  = list()
g_target = 0

def solution(numbers, target):
    answer = 0
    
    global number_size
    number_size = len(numbers)
    
    global g_numbers
    g_numbers = numbers
    
    global g_target
    g_target = target
    
    DFS(0)
    
    return count