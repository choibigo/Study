import input_setting

print(*sorted({input() for _ in range(int(input()))}, key=lambda x : (len(x), x)), sep="\n")