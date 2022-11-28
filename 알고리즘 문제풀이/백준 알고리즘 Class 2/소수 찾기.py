import input_setting

import sys
_ = int(input())
num_list = map(int, sys.stdin.readline().split(' '))

def is_prime(num):

    for i in range(2, int(num**0.5)+1):
        if num%i ==0:
            return False
    return True

answer = 0
for num in num_list:
    if num > 1 and is_prime(num):
        answer +=1

print(answer)