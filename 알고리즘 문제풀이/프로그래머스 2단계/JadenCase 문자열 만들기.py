def solution(s_list):
    
    answer = list()
    for s in s_list.split(" "):
        if len(s) != 0:
            answer.append(s[0].upper()+s[1:].lower())
            
    return " ".join(answer)

    