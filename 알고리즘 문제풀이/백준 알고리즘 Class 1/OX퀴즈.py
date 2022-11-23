import input_setting


for _ in range(int(input())):
    string = input()
    res = [0 for _ in range(len(string))]
    res[0] = 1 if string[0] == 'O' else 1
    for i in range(1, len(string)):
        if string[i] == 'O':
            res[i] = res[i-1]+1

    print(sum(res))