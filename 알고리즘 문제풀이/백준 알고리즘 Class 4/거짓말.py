import input_setting

# Need To Check
import sys
people_count, party_count = map(int, sys.stdin.readline().split(' '))
know_list = set(sys.stdin.readline().strip().split(' ')[1:])
party_list = [set(sys.stdin.readline().strip().split(' ')[1:]) for _ in range(party_count)]

for _ in range(party_count):
    for party in party_list:
        if party.intersection(know_list):
            know_list = know_list.union(party)

answer = 0
for party in party_list:
    if party.intersection(know_list):
        continue
    answer +=1

print(answer)