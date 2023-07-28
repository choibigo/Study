import input_setting

n = int(input())
meet_list = list()

for _ in range(n):
    meet_list.append(list(map(int, input().split())))

meet_list = sorted(meet_list, key=lambda x : (x[1],x[0]))

count=1
current_end_time=meet_list[0][1]

for meet in meet_list[1:]:
    if current_end_time <= meet[0]:
        current_end_time = meet[1]
        count+=1

print(count)
