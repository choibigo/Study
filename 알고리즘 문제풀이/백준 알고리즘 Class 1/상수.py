import input_setting

a, b = input().split()
print(max(int(a[::-1]), int(b[::-1])))
