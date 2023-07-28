import input_setting

import sys
a, b = map(int, input().split(' '))

key_id = dict()
key_name = dict()

for key in range(a):
    name = sys.stdin.readline().strip()
    key_id[key+1] = name
    key_name[name] = key+1

for _ in range(b):
    key = sys.stdin.readline().strip()

    if key.isdigit():
        print(key_id[int(key)])
    else:
        print(key_name[key])