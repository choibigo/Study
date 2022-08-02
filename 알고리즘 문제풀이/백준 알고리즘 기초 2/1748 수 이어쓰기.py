import input_setting

target_num = int(input())

total = 0
index_num = 1

while True:
    if index_num < target_num:
        total += (index_num//10) * 9 * len(list(str(index_num//10)))
        index_num *= 10

    elif index_num == target_num:
        total += (index_num//10) * 9 * len(list(str(index_num//10)))
        total += len(list(str(index_num)))
        break
    
    else:
        index_num = int(index_num / 10)
        total += (target_num - index_num + 1) * len(list(str(index_num)))
        break

print(total)