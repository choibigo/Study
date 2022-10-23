from collections import deque

def solution(priorities, location):
    
    priorities = deque(priorities)
    count = 1
    while len(priorities) != 1:
        pop_num = priorities.popleft()
    
        if pop_num >= max(priorities):
            if location == 0:
                return count
            location -= 1
            count +=1
        else:
            priorities.append(pop_num)
            
            if location == 0:
                location = len(priorities)-1
            else:
                location-=1
    
    return count
