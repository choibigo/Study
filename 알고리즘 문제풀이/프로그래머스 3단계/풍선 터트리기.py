def solution(a):
    result = [0 for _ in range(len(a))]
    
    l_min = float("inf")
    for i in range(len(a)):
        if l_min > a[i]:
            l_min = a[i]
            result[i] = 1
        
    r_min = float("inf")
    for i in range(len(a)-1, -1, -1):
        if r_min>a[i]:
            r_min = a[i]
            result[i] = 1

    return sum(result)