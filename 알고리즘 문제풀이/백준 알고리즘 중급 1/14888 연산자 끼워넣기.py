import input_setting

from itertools import permutations
import sys

max_val = -10000000001
min_val = 10000000001

num_count = int(input())
num_list = list(map(int, input().split()))
op_count = list(map(int, input().split()))

op_list = ["#"]
for op, count in zip(["+", "-", "*", "//"], op_count):
    op_list += [op] * count

check = [False] * num_count

def calculate_num(num1, num2, op):
    if op == "+":
        return num1+num2
    elif op == "-":
        return num1-num2
    elif op == "*":
        return num1*num2
    elif op == "//":
        if num1 < 0:
            num1 *= -1
            temp = num1//num2
            return -temp
        else:
            return num1//num2

def DFS(index, num):

    global max_val
    global min_val

    if index == num_count:
        if max_val < num:
            max_val = num
        if min_val > num:
            min_val = num

        return

    for i in range(1, num_count):
        if check[i] == False:
            check[i] = True
            next_num = calculate_num(num, num_list[index], op_list[i])
            DFS(index+1, next_num)
            check[i] = False

DFS(1,num_list[0])

print(max_val)
print(min_val)