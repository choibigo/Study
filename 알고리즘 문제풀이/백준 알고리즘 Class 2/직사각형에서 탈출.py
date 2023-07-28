import input_setting

x, y, w, h = map(int, input().split(' '))
print(min(abs(x-w), x, abs(y-h), y))