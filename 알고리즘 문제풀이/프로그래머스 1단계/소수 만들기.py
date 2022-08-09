def prime_num(num):
    for i in range(2, num):
        if num%i == 0:
            return False
        
    return True

def DFS(v, s):
    
    global answer
    
    if v == 3:
        if prime_num(sum(res)):
            answer+=1
        return
    
    for i in range(s, len_nums):
        res.append(g_nums[i])
        DFS(v+1, i+1)
        res.pop()

res = list()
len_nums = 0
answer = 0
g_nums = list()

import copy
def solution(nums):
    
    global g_nums
    g_nums = nums
    
    global len_nums
    len_nums = len(nums)
    
    DFS(0,0)
    
    return answer