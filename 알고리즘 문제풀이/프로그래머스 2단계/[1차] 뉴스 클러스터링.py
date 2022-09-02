from collections import Counter

def set_create(target_str):
    result = list()
    for i in range(len(target_str)-1):
        if target_str[i].isalpha() and target_str[i+1].isalpha():
            result.append(target_str[i].lower()+target_str[i+1].lower())

    return result

def Jacard(set1, set2):
    
    count1 = Counter(set1)
    count2 = Counter(set2)
    
    inter = list((count1 & count2).elements())
    union = list((count1 | count2).elements())
    
    if len(union) == 0:
        return 1
    
    return (len(inter)/len(union))
    
    
def solution(str1, str2):

    set1 = set_create(str1)
    set2 = set_create(str2)
    
    return int(Jacard(set1, set2)*65536)