import input_setting

n, case = map(int, input().split())
num_list = list(map(int, input().split()))

for _ in range(case):
    start, end = map(int, input().split())

    avg = sum(num_list[start-1:end])/(end-start+1)
    avg = round(avg, 2)
    print(avg)