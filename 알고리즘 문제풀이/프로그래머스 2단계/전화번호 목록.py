def solution(phone_book):

    phone_book.sort()
    
    for i in range(len(phone_book)-1):
        current_str = phone_book[i]
        next_str = phone_book[i+1]
        if len(current_str) < len(next_str):
            if current_str == next_str[0:len(current_str)]:
                print(current_str, next_str)
                return False
    
    return True