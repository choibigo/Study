import input_setting

for _ in range(int(input())):
    n, string = input().split()
    print("".join([s*int(n) for s in string]))