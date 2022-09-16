import re
from itertools import permutations

def solution(user_id_list, banned_id_list):
    
    ban = "/".join(banned_id_list).replace("*",".")
    answer = set()
    
    for temp in list(permutations(user_id_list, len(banned_id_list))):
        
        if re.fullmatch(ban, "/".join(temp)):
            temp = sorted(temp)
            answer.add("".join(temp))
            
    
    return len(answer)