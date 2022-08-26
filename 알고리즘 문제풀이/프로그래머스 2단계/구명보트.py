from collections import deque

def solution(people, limit):
    
    people.sort(reverse=True)
    print(people)
    people = deque(people)
    answer = 0
    temp = 0
    
    while len(people) > 1:
    
        if people[0] + people[-1] <= limit:
            people.popleft()
            people.pop()
            answer +=1
        else:
            people.popleft()
            answer+=1
    
    if len(people) != 0:
        answer += 1
    
    return answer