def primenumber(x):
    for i in range(2, int(x** 1/2)+1):
        if x % i == 0:
            return False
    return True

def DFS(v, numbers):
    global ch 

    if len(res) != 0:
        temp_num = int("".join(res))
        if temp_num >=2 and primenumber(temp_num):
            answer.add(temp_num)
    
    if v == len(numbers):
        return
    
    for i in range(0, len(numbers)):
        if ch[i] == -1:
            ch[i] = 0
            res.append(numbers[i])
            DFS(v+1, numbers)
            res.pop()
            ch[i] = -1
    
res = list()
ch = list()
answer = set()

def solution(numbers):

    global ch
    ch = [-1] * (len(numbers))
    
    numbers = list(numbers)
    DFS(0, numbers)
    
    return len(list(answer))