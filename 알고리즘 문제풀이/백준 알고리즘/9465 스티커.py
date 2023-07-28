import input_setting

case = int(input())

for _ in range(case):
    cols = int(input())

    if cols == 1:
        print(max(int(input()), int(input())))
    else:

        sticker = [list(map(int, input().split())) for _ in range(2)]
        res = [[0 for _ in range(cols)] for _ in range(2)]

        res[0][0] = sticker[0][0]
        res[1][0] = sticker[1][0]

        res[0][1] = sticker[0][1] + res[1][0]
        res[1][1] = sticker[1][1] + res[0][0]

        if cols > 2:        
            for col in range(2, cols):
                res[0][col] = max(res[0][col-2], res[1][col-2], res[1][col-1]) + sticker[0][col]
                res[1][col] = max(res[0][col-2], res[1][col-2], res[0][col-1]) + sticker[1][col]

        print(max(res[0][-1], res[1][-1]))