def solution(s):
    s = s[2:-2].split("},{")
    s = sorted(s, key = lambda x: len(x))
    
    answer = list()
    for t in s:
        for a in t.split(","):
            if int(a) not in answer:
                answer.append(int(a))
                break
        

    return answer
    