def solution(answers):
    
    sol1_list = [1,2,3,4,5]
    sol2_list = [2,1,2,3,2,4,2,5]
    sol3_list = [3,3,1,1,2,2,4,4,5,5]

    answer = [0,0,0]
    
    for i in range(len(answers)):
        if answers[i] == sol1_list[i%5]:
            answer[0] +=1
        
        if answers[i] == sol2_list[i%8]:
            answer[1] +=1
            
        if answers[i] == sol3_list[i%10]: 
            answer[2] +=1
    
    max_score = max(answer)
    
    result = list()
    
    for i in range(3):
        if answer[i] == max_score:
            result.append(i+1)
    
    return result