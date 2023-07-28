def solution(money):

    res1 = [0 for _ in range(len(money))]
    res1[0] = money[0]
    res1[1] = money[0]
    
    for i in range(2, len(money)-1):
        res1[i] = max(res1[i-1], res1[i-2]+money[i])
        
    res2 = [0 for _ in range(len(money))]
    res2[1] = money[1]
    
    for i in range(2, len(money)):
        res2[i] = max(res2[i-1], res2[i-2]+money[i])
        
    return max(max(res1), max(res2))