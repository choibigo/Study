import input_setting

n = int(input())
for _ in range(n):
    result = input()
    res = [0 for _ in range(len(result))]
    res[0] = 1 if result[0] == "O" else 0

    for i in range(1, len(result)):
        if result[i] == "O":
            res[i] = res[i-1] + 1

    print(sum(res))

print(ord('A'))