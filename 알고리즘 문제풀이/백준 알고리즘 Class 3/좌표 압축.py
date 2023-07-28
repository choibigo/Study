import input_setting


n = int(input())
num_list = list(map(int, input().split(" ")))
info = {num:i for i, num in enumerate(sorted(set(num_list)))}

for num in num_list:
    print(info[num], end=" ")