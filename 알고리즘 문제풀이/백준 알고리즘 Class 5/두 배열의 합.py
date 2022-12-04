import input_setting


# Need To Check
import bisect
t = int(input())
n = int(input())
Aarr = list(map(int, input().split()))
m = int(input())
Barr = list(map(int, input().split()))

Asum = list()
Bsum = list()

for i in range(n):
    Asum.append(Aarr[i])
    s = Aarr[i]
    for j in range(i+1, n):
        s += Aarr[j]
        Asum.append(s)

for i in range(m):
    Bsum.append(Barr[i])
    s = Barr[i]
    for j in range(i+1, m):
        s += Barr[j]
        Bsum.append(s)

Asum.sort()
Bsum.sort()
answer = 0

for i in range(len(Asum)):
    l = bisect.bisect_left(Bsum, t-Asum[i])
    r = bisect.bisect_right(Bsum, t-Asum[i])
    answer += r-l

print(answer)