import input_setting


# Need To Check
import sys
n = int(input())
numbers = list(map(int, sys.stdin.readline().split()))
m = int(input())

dp=[[0]*n for _ in range(n)]

for num_len in range(n):
    for start in range(n-num_len):
        end = start+num_len

        if start == end:
            dp[start][end] = 1

        elif numbers[start] == numbers[end]:
            if start+1 == end:
                dp[start][end] = 1

            elif dp[start+1][end-1] == 1:
                dp[start][end] = 1

for question in range(m):
    s, e = map(int, sys.stdin.readline().split())
    print(dp[s-1][e-1])