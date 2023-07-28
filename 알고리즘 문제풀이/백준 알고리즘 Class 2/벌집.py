import input_setting


n = int(input())
count=1
num=1
while True:
    if n <=num:
        print(count)
        break
    num += (6*count)
    count+=1