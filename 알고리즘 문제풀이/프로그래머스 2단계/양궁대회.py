# region 실패
max_count = -1
result = list()

def solution(n, info):
    
    res = list()
    res2 = [0] * 11
    global result
    
    def DFS(v, arow_count, apeach, lion):
           #과녁 점수, 화살 수, 어피치 점수, 라이언 점수
        
        global max_count
        global result
        
        if v == 10:
            if max_count < lion-apeach:
                max_count = lion-apeach
                result = res2[:]
        elif arow_count == 0:
            
            if max_count < lion-apeach:
                max_count = lion-apeach
                result = res2[:]
            
        else:
            # (10-v) 점수 어피치가 0일떄
            if info[v] == 0:
                res2[v] = 1
                DFS(v+1, arow_count-1, apeach, lion+(10-v))   
                res2[v] = 0
                
            else:
                # (10-v) 점수 라이언이 이기기
                if info[v]+1 <= arow_count: 
                    res2[v] = (info[v] + 1)
                    DFS(v+1, arow_count-(info[v] + 1), apeach-(10-v), lion+(10-v))   
                    res2[v] = 0

                # (10-v)점수 라이언이 지기
                DFS(v+1, arow_count, apeach, lion)   
            
    apeach_total = 0
    for score, a in enumerate(info):
        if a != 0 :
            apeach_total += (10-score)
        
    DFS(0, n, apeach_total, 0)
    
    if len(result) == 0:
        return [-1]
    else:
        if sum(result) != n:
            result[-1] = n - sum(result)
            
        return result
# endregion


from itertools import combinations_with_replacement

def solution(n, apeach_list):
    
    score_list = [x for x in range(0, 11)]
    max_count = 0
    max_list = list()
    
    for candidate in list(combinations_with_replacement(score_list, n)):
        lion_list = [0] * 11

        apeach = 0
        lion = 0
        
        for score in candidate:
            lion_list[(10 - score)] += 1
        
        for score, (apeach_shot, lion_shot) in enumerate(zip(apeach_list, lion_list)):
            if apeach_shot == lion_shot == 0:
                continue
            
            if apeach_shot >= lion_shot:
                apeach += (10-score)
            else:
                lion += (10-score)
                    
        if max_count < lion-apeach:
            max_count = lion-apeach
            max_list = lion_list[:]
    
    if len(max_list) == 0:
        return [-1]
    else:
        return max_list
        
    
    
    return 1