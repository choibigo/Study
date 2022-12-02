import input_setting

from collections import deque
n, k = map(int, input().split())
num_list = deque([i for i in range(1, n+1)])

idx = 0
result = ''
while len(num_list):
    idx +=1
    pop_num = num_list.popleft()
    if idx==k:
        result += f'{str(pop_num)}, '
        idx=0
        continue

    num_list.append(pop_num)

print(f"<{result[:-2]}>")