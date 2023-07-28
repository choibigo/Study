import input_setting


import sys
from collections import deque

def func(num_list):
    try:
        status = 'F'
        for op in op_list:
            if op == 'D':
                if status=='F':
                    num_list.popleft()
                elif status=='B':
                    num_list.pop()
            elif op == 'R':
                if status=='F':
                    status='B'
                elif status=='B':
                    status = 'F'

        if status == 'B':
            num_list.reverse()
        print("[" + ",".join(num_list) + "]")

    except:
        print('error')
    

for _ in range(int(input())):
    op_list = sys.stdin.readline().strip()
    num_count = int(input())
    num_list = deque(sys.stdin.readline().rstrip()[1:-1].split(","))
    if num_count == 0:
        num_list = []
    func(num_list)