import input_setting


def prime_nums(n):
    is_prime = [1 for _ in range(n+1)]
    is_prime[0] = 0
    is_prime[1] = 0

    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(2, n//i+1):
                is_prime[i*j] = 0

    result = list()
    for i in range(n+1):
        if is_prime[i]:
            result.append(i)
    return result

n = int(input())
prime_list = prime_nums(n)
len_prime_list = len(prime_list)

left = 0
right = 0
sum = 0
result = 0
while True:
    if sum >= n:
        if sum == n:
            result +=1
        sum -= prime_list[left]
        left+=1

    elif right ==len_prime_list:
        break
    else:
        sum += prime_list[right]
        right+=1

print(result)