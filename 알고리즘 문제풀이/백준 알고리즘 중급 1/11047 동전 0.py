import input_setting

n, target = map(int, input().split())
coin_list = list()
for _ in range(n):
    coin_list.append(int(input()))

count = 0
for coin in coin_list[::-1]:
    if target == 0:
        break
    
    if coin <= target:
        div, mod = divmod(target, coin)
        target = mod
        count += div

print(count)