num_list = list(map(int, input().split(" ")))
num_list = list(map(lambda x : x**2, num_list))
print(sum(num_list)%10)