import input_setting

n = int(input())
dp = [0,1]

for i in range(2, n+1):
    min_value = float('inf')
    for j in range(1, int(i**0.5)+1):
        min_value = min(min_value, dp[i-j**2])
    dp.append(min_value+1)

print(dp[n])