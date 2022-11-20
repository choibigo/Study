from collections import deque

def solution(order):

    line_box = deque([i for i in range(1, len(order)+1)])
    sub_line = list()
    
    answer = 0
    for o in order:
        if not line_box and sub_line[-1] != o:
            break
            
        if line_box and line_box[0] == o:
            line_box.popleft()
            answer +=1
        else:
            if sub_line and sub_line[-1] == o:
                sub_line.pop()
                answer +=1
            else:
                while line_box:
                    line_pop = line_box.popleft()
                    if line_pop == o:
                        answer +=1
                        break
                    else:
                        sub_line.append(line_pop)
    return answer