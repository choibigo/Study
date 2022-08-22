from collections import deque

def solution(queue1, queue2):
    
    queue1_sum = sum(queue1)
    queue2_sum = sum(queue2)
    
    queue1 = deque(queue1)
    queue2 = deque(queue2)
    
    for i in range(len(queue1) * 3):
        if queue1_sum == queue2_sum:
            return i
        
        if queue1_sum > queue2_sum:
            pop_num = queue1.popleft()
            queue1_sum -= pop_num
            
            queue2.append(pop_num)
            queue2_sum += pop_num 
        
        else:
            pop_num = queue2.popleft()
            queue2_sum -= pop_num
            
            queue1.append(pop_num)
            queue1_sum += pop_num 
    
    return -1