
def solution(begin, end):
    
    answer = list()
    
    for i in range(begin, end+1):
        if i == 1:
            answer.append(0)
        else:
            temp = 1
            for j in range(2, int(i**(1/2))+1):
                mok = i//j
                
                if mok > 10 ** 7:
                    continue
                
                if i%j == 0:
                    temp = mok
                    break
            answer.append(temp)
                    
    return answer