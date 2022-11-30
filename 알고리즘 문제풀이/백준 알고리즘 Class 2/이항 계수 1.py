import input_setting

def func(args):
    def factorial(n):
        if n <2:
            return 1
        return factorial(n-1)*n
    n, k = args
    up = factorial(n)
    down = factorial(n-k)*factorial(k)
    print(up//down)

func(map(int, input().split()))