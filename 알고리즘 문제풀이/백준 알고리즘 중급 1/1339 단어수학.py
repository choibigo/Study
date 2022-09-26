import input_setting

count = int(input())
str_list = [input() for _ in range(count)]

alpha_info = dict()
alpha_num = dict()
for i in range(65, 91):
    alpha_info[chr(i)] = 0
    alpha_num[chr(i)] = -1

for s in str_list:
    s = s[::-1]
    for i in range(len(s)):
        alpha_info[s[i]] += 10**i

alpha_info = sorted(alpha_info.items(), key=lambda x : x[1], reverse=True)

for i, (key, value) in enumerate(alpha_info, 1):
    if value == 0:
        break
    alpha_num[key] = str(10-i)

answer = 0
for word in str_list:

    temp = ""
    for w in word:
        temp += alpha_num[w]

    answer += int(temp)

print(answer)