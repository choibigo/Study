import input_setting


for x,y in sorted([tuple(map(int,input().split())) for _ in range(int(input()))]):
    print(x, y)