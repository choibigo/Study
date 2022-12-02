import input_setting



a, b, v = map(int, input().split())
day, mod = divmod(v-a, a-b)
print(day+2 if mod else day+1)