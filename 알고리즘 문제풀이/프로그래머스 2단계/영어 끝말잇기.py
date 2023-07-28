def solution(n, words):
    
    speak_list = list()
    
    for index, word in enumerate(words):
        
        if len(speak_list) == 0:
            speak_list.append(word)
        else:
            if word in speak_list:
                temp1 = index//n + 1
                temp2 = index%n + 1
                return [temp2, temp1]
            else:
                if speak_list[-1][-1] == word[0]:
                    speak_list.append(word)
                else:
                    temp1 = index//n + 1
                    temp2 = index%n + 1
                    return [temp2, temp1]
                    
    return [0,0]
                