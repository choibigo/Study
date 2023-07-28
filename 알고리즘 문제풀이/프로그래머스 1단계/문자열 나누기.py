from collections import deque

def solution(s):

    s = deque(s)
    result = 0
    while s:
        x = s.popleft()
        x_count = 1
        not_x_count = 0
        result+=1
        while s:
            pop_s = s.popleft()
            if pop_s == x:
                x_count+=1
            else:
                not_x_count +=1
            if x_count == not_x_count:
                break
    
    return result                