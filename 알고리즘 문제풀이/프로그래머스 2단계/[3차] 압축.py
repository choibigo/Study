def solution(msg):
    answer = list()
    next_num = 27
    res = {chr(i+64):i for i in range(1, 27)}
    
    index = 0
    word = ""
    while index < len(msg):
        
        word += msg[index]
        if word not in res:
            res[word] = next_num
            next_num +=1
            answer.append(res[word[:-1]])
            word = msg[index]
        index += 1
    
    answer.append(res[word])
    
    return answer