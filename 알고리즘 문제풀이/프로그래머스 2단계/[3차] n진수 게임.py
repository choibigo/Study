def n_num(num, n):
    
    res = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F"]
    
    if num <n:
        return res[num]
    
    return n_num(num//n, n) + res[num%n]
    


def solution(n, t, m, p):
    answer = ''

    num_list = ""
    
    for i in range(t*m):
        num_list += (n_num(i, n))
    
    for i in range(t):
        answer += (num_list[i*m + (p-1)])

    return answer