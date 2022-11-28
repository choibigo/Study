import input_setting

def func(num):
    for n in range(1, num):
        if n+sum(list(map(int, str(n)))) == num:
            print(n)
            return
    print(0)

func(int(input()))