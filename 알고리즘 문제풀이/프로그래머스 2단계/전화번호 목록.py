def solution(phone_book):

    phone_book.sort()
    for a, b in zip(phone_book, phone_book[1:]):
        if a == b[:len(a)]:
            return False
    
    return True