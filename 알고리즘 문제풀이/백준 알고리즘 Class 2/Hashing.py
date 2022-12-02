import input_setting


L = int(input())
list_s = input()
sum = 0
for i in range(L):
    sum += (ord(list_s[i]) - 96) * (31 ** i)
    
sum %= 1234567891
print(sum)