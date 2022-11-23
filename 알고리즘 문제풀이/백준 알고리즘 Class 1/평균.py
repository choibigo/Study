import input_setting

n = int(input())
score_list = list(map(int, input().split()))
max_score = max(score_list)
score_list = list(map(lambda x: x/max_score*100, score_list))

print(sum(score_list)/n)