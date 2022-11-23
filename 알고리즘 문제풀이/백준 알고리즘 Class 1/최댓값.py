
import input_setting

num_list = list()
while True:
    try:
        num_list.append(int(input()))
    except:
        break

max_value = max(num_list)

print(max_value)
print(num_list.index(max_value)+1)