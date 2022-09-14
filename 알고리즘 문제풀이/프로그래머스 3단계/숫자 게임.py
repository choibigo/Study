# region My Solution
import heapq

def solution(A_list, B_list):
    
    A_list.sort()
    
    heapq.heapify(B_list)
    
    answer = 0
    index = 0
    while index < len(A_list):
        a = A_list[index]
        
        if len(B_list) == 0:
            break
            
        while B_list:
            pop_num = heapq.heappop(B_list)
            if pop_num == a:
                break
            elif pop_num > a:
                answer +=1
                index += 1
                break
        
    
    return answer
# endregion

# region Other Solution

def solution(A_list, B_list):
    A_list.sort()
    B_list.sort()
    
    a_index = 0
    answer = 0
    for i in range(len(B_list)):
        if B_list[i] > A_list[a_index]:
            answer +=1
            a_index += 1
        
    return answer
# endregion


