import input_setting

# region Permutations
# from itertools import permutations

# count = int(input())
# sign_list = input().split()

# result_list = list()

# for num_list in list(permutations(list(x for x in range(10)), count+1)):
    
#     flag = True
#     for i, sign in enumerate(sign_list):
#         if sign == "<":
#             if num_list[i] < num_list[i+1]:
#                 continue
#             else:
#                 flag = False
#                 break

#         elif sign == ">":
#             if num_list[i] > num_list[i+1]:
#                 continue
#             else:
#                 flag = False
#                 break
#     if flag:
#         result_list.append(num_list)

# print("".join(map(str, result_list[-1])))
# print("".join(map(str, result_list[0])))
# endregion

# region 재귀

def condition(num1, num2, op):
    if op == "<":
        if num1<num2:
            return True

    if op == ">":
        if num1>num2:
            return True
    
    return False

count = int(input())
sign_list = input().split()
num_list = [x for x in range(10)]
check = [False] * 10

result_list = list()
def DFS(index, num):
    if index == count+1:
        result_list.append(num)
        return 
    
    for i in range(10):
        if check[i] == True : continue

        if index==0 or condition(int(num[index-1]), i, sign_list[index-1]):
            check[i] = True
            DFS(index+1, num+str(i))
            check[i] = False

DFS(0, "")

print(result_list[-1])
print(result_list[0])

# endregion