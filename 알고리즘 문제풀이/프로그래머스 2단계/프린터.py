from collections import deque

def solution(priorities, location):
    
    priorities = deque(priorities)
    
    
    target_location = location
    order = 1
    
    while len(priorities) != 1:
        pop_num = priorities.popleft()
        
        if pop_num >= max(priorities):
            if target_location == 0:
                return order
            else:
                order +=1
                target_location -= 1
        else: 
            priorities.append(pop_num)
            
            if target_location == 0:
                target_location = len(priorities)-1
            else:
                target_location -=1
                
        print(priorities, target_location, order)
        
    
    return order
 
