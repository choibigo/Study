from itertools import zip_longest

def solution(participant, completion):
    
    participant.sort()
    completion.sort()
    
    for p, c in zip_longest(participant, completion):
        if p !=c :
            return p
