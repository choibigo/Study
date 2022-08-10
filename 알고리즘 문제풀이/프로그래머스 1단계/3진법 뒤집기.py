def solution(N, stages):
    
    # 실패율 = 클리어 못한 수 / 스테이지에 도달한 수
    
    stage_unclear = [0] * (N+2)
    stage_unclear[0] = -999
    
    for s in stages:
        stage_unclear[s] += 1
    
    stage_arrive = [0] * (N+2)
    stage_arrive[0] = -999
    stage_arrive[1] = len(stages)
    
    for i in range(2, N+1):
        stage_arrive[i] = stage_arrive[i-1] - stage_unclear[i-1]
    
    result = dict()
    
    for i in range(1, N+1):
        
        if stage_arrive[i] == 0:
            result[i] = 0
        
        else:
            result[i] = (stage_unclear[i] / stage_arrive[i])
    
    result = sorted(result, key = lambda x : result[x], reverse = True)
    
    return result