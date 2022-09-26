import input_setting

num = int(input())

if num == 1:
    print(0)
elif num <=3:
    print(1)
else:
    res = [0 for _ in range(num+1)]

    res[1] = 0
    res[2] = 1
    res[3] = 1

    for i in range(4, num+1):
        temp = list()

        # -1
        temp.append(res[i-1])
        # %2
        if i % 2 == 0:
            temp.append(res[i//2])

        if i % 3 == 0:
            temp.append(res[i//3])
        res[i] = min(temp)+1

    print(res[num])