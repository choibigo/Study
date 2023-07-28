import input_setting

import sys
from collections import defaultdict
info_count, result_count = map(int, sys.stdin.readline().split(' '))

info = defaultdict(lambda x :'')
for _ in range(info_count):
    key, value = sys.stdin.readline().strip().split(' ')
    info[key] = value

for _ in range(result_count):
    key = sys.stdin.readline().strip()
    print(info[key])