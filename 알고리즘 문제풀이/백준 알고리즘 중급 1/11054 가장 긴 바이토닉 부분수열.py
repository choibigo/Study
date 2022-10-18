import input_setting

n = int(input())
num_list = [*map(int, input().split())]

res_1 = [1]*n
res_2 = [1]*n

for index in range(n):
    for j in range(index):
        if num_list[j]<num_list[index]:
            res_1[index] = max(res_1[index], res_1[j]+1)

num_list = num_list[::-1]

for index in range(n):
    for j in range(index):
        if num_list[j]<num_list[index]:
            res_2[index] = max(res_2[index], res_2[j]+1)

temp = [r1+r2 for r1, r2 in zip(res_1, res_2[::-1])]
print(max(temp)-1)