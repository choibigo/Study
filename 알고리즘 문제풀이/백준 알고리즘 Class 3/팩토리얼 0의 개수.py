import input_setting


def factorial(num):
    if num == 1:
        return num
    return num*factorial(num-1)

num = int(input())
if num ==0:
    print(0)
else:
    for i, n in enumerate(str(factorial(num))[::-1]):
        if int(n)>0:
            print(i)
            break