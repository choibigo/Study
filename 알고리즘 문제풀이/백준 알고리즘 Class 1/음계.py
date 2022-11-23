import input_setting


def func(num_list):
    offset = num_list[1]-num_list[0]

    if abs(offset) != 1:
        return "mixed"

    for i in range(2, len(num_list)):
        if num_list[i] - num_list[i-1] != offset:
            return "mixed"

    if offset==1:
        return "ascending"
    else:
        return "descending"


if __name__ =="__main__":
    num_list = list(map(int, input().split(' ')))
    print(func(num_list))
