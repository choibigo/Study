import input_setting


import sys
n = int(input())
num_list = sorted(list(map(int, input().split())))
result = num_list[-3:]
min_sum = abs(sum(result))
for i in range(n-2):
    left = i+1
    right = n-1

    while left<right:
        sum_value = num_list[i] + num_list[left] + num_list[right]
        if min_sum > abs(sum_value):
            min_sum = abs(sum_value)
            result = [num_list[i], num_list[left], num_list[right]]

        if sum_value < 0:
            left +=1
        elif sum_value == 0:
            print(*result)
            sys.exit()
        else:
            right -=1

print(*result)