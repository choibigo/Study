def solution(id_list, report, k):
    
    from collections import defaultdict
    
    report_dict = defaultdict(list)
    
    for r in report:
        people, bad = r.split(" ")
        if people not in report_dict[bad]:
            report_dict[bad].append(people)
        
    answer_dict = defaultdict(int)
    
    for key, item in report_dict.items():
        if len(item) >= k:
            for name in item:
                answer_dict[name] +=1
    
    answer = []
    
    for id in id_list:
        if id in answer_dict:
            answer.append(answer_dict[id])
        else:
            answer.append(0)
    
    
    return answer