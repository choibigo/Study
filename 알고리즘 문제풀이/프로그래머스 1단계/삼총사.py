# region itertools
# from itertools import combinations

# def solution(number):
    
#     answer = 0
#     for temp in list(combinations(number, 3)):
#         if sum(temp) == 0:
#             answer += 1
            
#     return answer

# endregion

# region custom combination
def solution(number):
    
    count = 0
    def DFS(v, s, sum_val):
        nonlocal count
        
        if v == 3:
            if sum_val == 0:
                count +=1
            return 
    
        for i in range(s, len(number)):
            DFS(v+1, i+1, sum_val + number[i])
            
    DFS(0,0,0)
    return count

# endregion


