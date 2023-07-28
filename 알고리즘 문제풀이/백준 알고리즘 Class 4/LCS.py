import input_setting

# Need To Check
import sys
str1 = "#"+sys.stdin.readline().strip()
str2 = "#"+sys.stdin.readline().strip()

res = [[0 for _ in range(len(str1))] for _ in range(len(str2))]

for row in range(1, len(str2)):
    for col in range(1, len(str1)):
        if str1[col] == str2[row]:
            res[row][col] = res[row-1][col-1] + 1
        else:
            res[row][col] = max(res[row-1][col], res[row][col-1])

print(res[-1][-1])