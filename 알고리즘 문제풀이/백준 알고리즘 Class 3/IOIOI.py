import input_setting

import sys
n = int(input())
str_count = int(input())
target = sys.stdin.readline().strip()

i = 0
count = 0
answer = 0

while i < str_count-1:
    if target[i:i+3] == 'IOI':
        i+=2
        count+=1

        if count == n:
            count -= 1
            answer +=1

    else:
        i+=1
        count = 0

print(answer)

# Need TO Check