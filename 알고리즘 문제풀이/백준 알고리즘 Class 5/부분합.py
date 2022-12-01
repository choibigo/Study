import input_setting


import sys
n, s = map(int, input().split())
num_list = list(map(int, sys.stdin.readline().split()))

def func(num_list, n, s):
    if num_list[0] >=s:
        return 1

    result = n+1
    left=0
    right=1
    sum_value = num_list[0] + num_list[1]

    while True:
        if left==right:
            if sum_value >=s:
                return 1
            right +=1
            if right == n:
                break
            sum_value += num_list[right]

        else:
            if sum_value >=s:
                result = min(result, right-left+1)
                sum_value -= num_list[left]
                left+=1
            else:
                right +=1
                if right ==n:
                    break
                sum_value += num_list[right]

    if result == n+1:
        return 0
    return result

print(func(num_list, n, s))
