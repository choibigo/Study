def solution(info_list, query_list):
    
    table = dict()
    
    for lang in ['cpp', 'java', 'python', "-"]:
        for job in ['backend', 'frontend', "-"]:
            for career in ['junior', 'senior', "-"]:
                for food in ['chicken', 'pizza', "-"]:
                    table[lang + job + career + food] = []
    
    for idx, info in enumerate(info_list):
        temp = info.split(" ")
        
        for language in [temp[0], "-"]:
            for area in [temp[1], "-"]:
                for carrer in [temp[2], "-"]:
                    for food in [temp[3], "-"]:
                        key = language+area+carrer+food
                        if key in table:
                            table[key].append(int(temp[4]))
                            
    for key in table.keys():
        table[key].sort()
    
    answer = list()
    for query in query_list:
        query = query.replace(" and ", " ")
        lan, area, car, food, score = query.split(" ")
        
        candidate = table[lan+area+car+food]
        
        left = 0
        right = len(candidate)-1
        temp = len(candidate)
        
        while left<=right:
            mid = (left + right)//2
            
            if int(score) <= candidate[mid]:
                temp = mid
                right = mid - 1
            else:
                left = mid + 1
        
        answer.append(len(candidate) - temp)
    return answer