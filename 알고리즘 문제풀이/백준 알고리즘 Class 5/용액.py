import input_setting


import sys
n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

left = 0
right = n-1
min_value = abs(num_list[0] + num_list[-1])
r_left = 0
r_right = n-1

while left<right:
    sum_value = num_list[left]+num_list[right]
    if sum_value == 0:
        r_left = left
        r_right= right
        break

    if abs(sum_value) < min_value:
        r_left = left
        r_right= right
        min_value = abs(sum_value)

    if sum_value < 0:
        left+=1
    else:
        right-=1

print(num_list[r_left], num_list[r_right])