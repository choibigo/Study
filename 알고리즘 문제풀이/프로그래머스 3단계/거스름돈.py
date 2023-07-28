def solution(n, money):

    res = [0] * (n+1)
    for m in money:
        res[m] +=1
        for i in range(m+1, n+1):
            res[i] += res[i-m]
            
    return res[n]