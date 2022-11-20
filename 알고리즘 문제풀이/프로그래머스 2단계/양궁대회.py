from collections import deque

def solution(n, info):

    apeach = 0
    for score, i in enumerate(info):
        if i:
            apeach += 10-score
            
    status = [0 for _ in range(11)]
    nodes = deque([[n, 10, 0, apeach, status]])
    max_score = -1
    result = list()
    
    while nodes:
        remain_n, score, lion_score, apeach_score, status = nodes.popleft()
        
        if remain_n == 0:
            if lion_score > apeach_score and max_score <= lion_score-apeach_score:
                max_score = lion_score-apeach_score
                result = status
        else:
            if score == 0:
                nstatus = status[:]
                nstatus[10] = remain_n
                nodes.append([0, score-1, lion_score, apeach_score, nstatus[:]])

            elif score >0:
                if info[10-score]==0:
                    nstatus = status[:]
                    nstatus[10-score] = 1
                    nodes.append([remain_n-1, score-1, lion_score+score, apeach_score, nstatus])

                elif remain_n > info[10-score]:
                    nstatus = status[:]
                    nstatus[10-score] = info[10-score]+1
                    nodes.append([remain_n-(info[10-score]+1), score-1, lion_score+score, apeach_score-score, nstatus])

                nodes.append([remain_n, score-1, lion_score, apeach_score, status[:]])
    
    return result if result else [-1]
        
        