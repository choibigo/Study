
def solution(survey, choices):
    
    # 1번째 비동의 관련 
    # 2번째 동의 관련
    
    result = dict()
    result['R'] = 0
    result['T'] = 0
    result['C'] = 0
    result['F'] = 0
    result['J'] = 0
    result['M'] = 0
    result['A'] = 0
    result['N'] = 0
    
    for i in range(len(survey)):
        not_check, ok_check = list(survey[i])
        score = choices[i]
        
        if 1<=score<=3:
            result[not_check] += int(4-score)
        
        elif 5<=score<=7:
            result[ok_check] += int(score-4)
        
    print(result)
    
    RT = "R" if result['R'] >= result['T'] else "T"
    CF = "C" if result['C'] >= result['F'] else "F"
    JM = "J" if result['J'] >= result['M'] else "M"
    AN = "A" if result['A'] >= result['N'] else "N"
    
    return RT+CF+JM+AN