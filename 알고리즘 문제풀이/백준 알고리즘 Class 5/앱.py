import input_setting

# Need To Check
import sys
n, memory = map(int, input().split())
mem_list = [0]+list(map(int, sys.stdin.readline().split()))
cost_list = [0]+list(map(int, sys.stdin.readline().split()))

dp = [[0 for _ in range(sum(cost_list)+1)] for _ in range(n+1)]

result = float('inf')
for i in range(1, n+1):
    for j in range(1, len(dp[0])):
        cost = cost_list[i]
        mem = mem_list[i]

        if cost>j:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(mem+dp[i-1][j-cost], dp[i-1][j])

        if dp[i][j]>= memory:
            result = min(result, j)

print(result)