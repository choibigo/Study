import input_setting

def check(target):
    for t in target:
        if t in ['(', '[']:
            stack.append(t)
        elif t == ')':
            if len(stack):
                if stack[-1]=='(':
                    stack.pop()
                else:
                    return "no"
            else:
                return "no"

        elif t == ']':
            if len(stack):
                if stack[-1]=='[':
                    stack.pop()
                else:
                    return "no"
            else:
                return "no"

    return 'yes' if len(stack)==0 else 'no'

while True:
    stack = list()
    target = list(input())

    if target == ['.']:
        break

    print(check(target))