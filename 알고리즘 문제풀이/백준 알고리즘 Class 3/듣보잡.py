import input_setting

import sys
a_count, b_count = map(int, input().split())

a = set()
b = set()

for _ in range(a_count):
    a.add(sys.stdin.readline())

for _ in range(b_count):
    b.add(sys.stdin.readline())

result = a.intersection(b)
result = sorted(result)

print(len(result))
print(*result, sep="")

