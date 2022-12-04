import input_setting


import sys
str1 = sys.stdin.readline().strip()
len_str1 = len(str1)
str2 = sys.stdin.readline().strip()
len_str2 = len(str2)

res = [[[0, ""] for _ in range(len_str1+1)] for _ in range(len_str2+1)]

for s1 in range(1, len_str1+1):
    for s2 in range(1, len_str2+1):
        if str1[s1-1] == str2[s2-1]:
            res[s2][s1] = [res[s2-1][s1-1][0] + 1, res[s2-1][s1-1][1]+str1[s1-1]]
        else:
            if res[s2-1][s1][0] > res[s2][s1-1][0]:
                res[s2][s1] = res[s2-1][s1]
            else:
                res[s2][s1] = res[s2][s1-1]

if res[-1][-1][0]:
    print(*res[-1][-1], sep='\n')
else:
    print(0)
