import input_setting

import sys
target_str = sys.stdin.readline().strip()
boom_str = sys.stdin.readline().strip()
len_boom = len(boom_str)
stack = list()

# region
# idx = 0
# for t in target_str:
#     if t == boom_str[idx]:
#         if idx == len_boom-1:
#             for _ in range(len_boom-1):
#                 stack.pop()
#             if stack:
#                 idx = stack[-1][1]
#         else:
#             idx+=1
#             stack.append([t,idx])
#     else:
#         idx = 0
#         if t == boom_str[0]:
#             idx = 1
#         stack.append([t,idx])

# if stack:
#     result = ""
#     for s, _ in stack:
#         result+=s
#     print(result)
# else:
#     print("FRULA")

# endregion 

# region 효율 UP
for t in target_str:
    stack.append(t)
    if "".join(stack[-len_boom:]) == boom_str:
        for _ in range(len_boom):
            stack.pop()

if stack:
    print("".join(stack))
else:
    print('FRULA')
# endregion