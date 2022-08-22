from collections import deque

def solution(progresses, speeds):
    
    # progresses = [p+s for p,s in zip(progresses, speeds)]
    
    answer = list()
    
    progresses = deque(progresses)
    speeds = deque(speeds)
    
    while (progresses):
        count = 0
        while progresses and progresses[0] >= 100:
            progresses.popleft()
            speeds.popleft()
            count +=1
            
        if count:
            answer.append(count)
            
        progresses = deque([p+s for p,s in zip(progresses, speeds)])
    
    return answer