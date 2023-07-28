import input_setting



def func1(num1, num2):
    while num2:
        num1, num2 = num2, num1%num2
    return num1
    
num1, num2 = map(int, input().split(' '))
result1 = func1(num1, num2)
result2 = num1*num2 // result1

print(result1)
print(result2)