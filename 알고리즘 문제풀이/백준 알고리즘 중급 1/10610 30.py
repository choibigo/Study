import input_setting

num_list = sorted(list(map(int, input())), reverse=True)
if num_list[-1] !=0:
    print(-1)
elif sum(num_list)%3 != 0:
    print(-1)
else:
    print("".join(list(map(str, num_list))))

