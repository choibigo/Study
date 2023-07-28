import input_setting

n = int(input())
target_list = [input() for _ in range(n)]
answer = list()

for i in range(len(target_list[0])):

    temp = target_list[0][i]
    flag = True

    for j in range(1, len(target_list)):
        if target_list[j][i] != temp:
            flag = False
            break

    if flag:
        answer.append(temp)
    else:
        answer.append('?')

print("".join(answer))