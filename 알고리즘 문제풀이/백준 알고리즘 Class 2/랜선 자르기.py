import input_setting

k, n = map(int, input().split(" "))
lan_list = [int(input()) for _ in range(k)]

left = 1
right= max(lan_list)
while left<=right:
    mid = (left+right)//2
    count = 0
    for l in lan_list:
        count += (l//mid)

    if count < n:
        right = mid-1
    else:
        answer = mid
        left = mid+1

print(answer)