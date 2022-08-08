import input_setting

import sys

def add(num):
    res[num] = 1

def remove(num):
    res[num] = 0

def check(num):
    print(res[num])

def toggle(num):
    if res[num] == 1:
        res[num] = 0
    else:
        res[num] = 1

def all():
    global res
    res = [1] * (21)

def empty():
    global res
    res = [0] * (21)

res = [0] * (21)
n = int(input())

for i in range(n):
    temp = sys.stdin.readline().split()
    
    if temp[0] == "add":
        add(int(temp[1]))
    
    if temp[0] == "remove":
        remove(int(temp[1]))
    
    if temp[0] == "check":
        check(int(temp[1]))

    if temp[0] == "toggle":
        toggle(int(temp[1]))
    
    if temp[0] == "all":
        all()
    
    if temp[0] == "empty":
        empty()

