
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
def solution(n, lost, reserve):

    lost.sort()
    reserve.sort()

    res = [0] * (n+1)
    res[0] = -999

    for r in reserve:
        if r in lost:
            lost.remove(r)
        else:
            res[r] = 1

    for l in lost:
        if res[l] == 1:
            res[l] = 0
        elif res[l-1] == 1:
            res[l-1] = 0
            res[l] = 0
        elif l+1 <= n and res[l+1] == 1:
            res[l+1] = 0
            res[l] = 0
        else:
            res[l] = -1

    answer = n - res.count(-1)

    return answer