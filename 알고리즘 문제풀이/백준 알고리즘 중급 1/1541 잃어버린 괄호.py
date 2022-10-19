import input_setting

express = [sum(list(map(int,e.split("+")))) for e in input().split("-")]
print(express[0] - sum(express[1:]))

