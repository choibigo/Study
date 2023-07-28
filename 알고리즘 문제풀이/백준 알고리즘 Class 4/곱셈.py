import input_setting

a, b, c = map(int, input().split(' '))

def DFS(a, b, c):
    if b == 1:
        return a%c
    elif b%2 == 0:
        return (DFS(a, b//2, c)**2)%c
    else:
        return ((DFS(a,b//2,c)**2*a)%c)
print(DFS(a,b,c))