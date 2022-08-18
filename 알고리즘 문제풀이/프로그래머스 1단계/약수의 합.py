def solution(n):
    
    
    temp = list()
    
    for i in range(1, n+1):
        if n%i == 0:
            if i in temp:
                break
            if i != n//i:
                temp.append(i)
                temp.append(n//i)
            else:
                temp.append(i)
    
    return sum(temp)