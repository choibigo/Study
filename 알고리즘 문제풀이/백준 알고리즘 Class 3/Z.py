import input_setting

n, row, col = map(int, input().split())
n = 2**n
answer = 0

while n >2:
    n //= 2
    if 0<=row<n and 0<=col<n:
        answer += 0
    elif 0<=row<n and n<=col:
        col -=n
        answer += (n**2)
    elif n<=row and 0<=col<n:
        row -=n
        answer += (n**2)*2
    elif n<=row and n<=col:
        col-=n
        row-=n
        answer += (n**2)*3

if row and col:
    answer+=3
elif col:
    answer+=1
elif row:
    answer+=2

print(answer)

