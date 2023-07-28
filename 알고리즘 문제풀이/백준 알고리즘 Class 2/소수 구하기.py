import input_setting

def is_prime(num):
    for i in range(2, int(num**0.5)+1):
        if num%i ==0:
            return False
    return True

low, high = map(int, input().split(" "))

for num in range(low, high+1):
    if num!= 1 and is_prime(num):
        print(num)
