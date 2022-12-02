import input_setting

count = int(input())
idx = 0
num = 665

while True:
    if '666' in str(num):
        idx +=1
        if idx==count:
            print(num)
            break
    num+=1
