def solution(origin_elements):

    elements = origin_elements*2
    answer = set()
    
    for size in range(1, len(origin_elements)+1):
        for i in range(len(elements)):
            answer.add(sum(elements[i:i+size]))
            
    return len(answer)
        