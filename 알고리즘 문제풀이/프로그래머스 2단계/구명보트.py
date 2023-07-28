from collections import deque

def solution(people, limit):
    people = deque(sorted(people, key = lambda x : -x))
    
    count = 0
    while len(people) > 1:
        
        if people[0] + people[-1] <= limit:
            people.popleft()
            people.pop()
        else:
            people.popleft()
        count +=1
    
    return count+len(people)