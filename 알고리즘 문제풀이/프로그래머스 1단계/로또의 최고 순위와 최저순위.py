def rank(num):
    if num == 6:
        return 1
    elif num == 5:
        return 2
    elif num ==4:
        return 3
    elif num == 3:
        return 4
    elif num == 2:
        return 5
    else:
        return 6

def solution(lottos, win_nums):
    
    for win in win_nums:
        if win in lottos:
            lottos.remove(win)
    
    zero_count = 0
    for lotto in lottos:
        if lotto == 0:
            zero_count +=1
    
    answer = []
    answer.append(rank(6-(len(lottos) - zero_count)))
    answer.append(rank(6-(len(lottos))))
    
    return answer