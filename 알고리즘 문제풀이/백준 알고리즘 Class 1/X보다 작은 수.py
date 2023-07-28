import input_setting

count, n = map(int ,input().split(" "))
num_list = list(map(int, input().split(" ")))
num_list = list(filter(lambda x : x<n, num_list))
print(*num_list, sep=" ")