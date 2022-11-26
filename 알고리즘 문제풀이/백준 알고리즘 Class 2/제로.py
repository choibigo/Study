import input_setting

stack = list()
for _ in range(int(input())):
    num = int(input())

    if num == 0:
        if len(stack):
            stack.pop()
    else:
        stack.append(num)

print(sum(stack))