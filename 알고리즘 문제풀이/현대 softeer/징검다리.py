import input_setting

n = int(input())
stone_list = list(map(int, input().split()))

res = [1]*n

for index in range(n):
    for j in range(index):
        if stone_list[index]>stone_list[j]:
            res[index] = max(res[index], res[j]+1)

print(max(res))