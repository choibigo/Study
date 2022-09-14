def solution(genres, plays):
    
    music = dict()
    plays_sum = dict()
    
    for i, (g, p) in enumerate(zip(genres, plays)):
        
        if g not in plays_sum:
            plays_sum[g] = p
        else:
            plays_sum[g] +=p
            
        if g not in music:
            music[g] = [[p, i]]
        else:
            music[g].append([p,i])
    
    answer = list()
    for g, _ in sorted(plays_sum.items(), key = lambda x : -x[1]):
        for p, i in sorted(music[g], key = lambda x : (-x[0], x[1]))[:2]:
            answer.append(i)
    
    return answer