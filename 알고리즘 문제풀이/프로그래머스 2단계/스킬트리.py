def solution(skill, skill_trees):
    
    skill = list(skill)
    count = 0
    for skill_tree in skill_trees:
        index = 0
        check = True
        for s in skill_tree:
            if s in skill:
                if skill[index] == s:
                    index +=1
                else:
                    check = False
                    break
        if check:
            count+=1
            
    
    return count