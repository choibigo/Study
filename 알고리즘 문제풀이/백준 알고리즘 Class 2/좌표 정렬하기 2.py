import input_setting


for x,y in sorted([tuple(map(int,input().split())) for _ in range(int(input()))], key=lambda x:(x[1], x[0])):
    print(x, y)