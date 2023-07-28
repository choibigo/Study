def solution(babbling):

    result = 0
    for baby in babbling:
        for t in ["aya", "ye", "woo", "ma"]:
            if t*2 not in baby:
                baby = baby.replace(t, ' ')
        if baby.strip()=='':
            result +=1
    
    return result