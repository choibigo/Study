import input_setting

K, P, N = map(int, input().split())

def DFS(x, y):

    if y == 1:
        return x

    elif y%2==0:
        a = DFS(x, y/2)

        return a*a % 1000000007

    else:
        b = DFS(x, (y-1)/2)

        return b*b*x % 1000000007


print(K * DFS(P, N*10))