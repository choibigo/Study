from collections import Counter

def jacard(set1, set2):
    
    count1 = Counter(set1)
    count2 = Counter(set2)
    
    inter = list((count1 & count2).elements())
    union = list((count1 | count2).elements())
    
    if len(union) == 0:
        return 1
    
    return (len(inter)/len(union))
    

def solution(str1, str2):
    
    str1 = [(str1[i]+str1[i+1]).lower() for i in range(len(str1)-1) if str1[i].isalpha() and str1[i+1].isalpha()]
    str2 = [(str2[i]+str2[i+1]).lower() for i in range(len(str2)-1) if str2[i].isalpha() and str2[i+1].isalpha()]

    return int(jacard(str1, str2)*65536)
        
    
