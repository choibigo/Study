import input_setting

import sys
for _ in range(int(input())):
    info = dict()
    for _ in range(int(input())):
        item, category = sys.stdin.readline().strip().split(' ')

        if category in info:
            info[category].append(item)
        else:
            info[category] = [item]

    answer = 1
    for key, value in info.items():
        answer *= (len(value)+1)
    
    print(answer-1)