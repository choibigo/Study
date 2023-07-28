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




def solution(skill, skill_trees):
    
    skill = list(skill)
    count = 0
    
    for tree in skill_trees:
        temp_skill = skill[:]
        
        for t in tree:
            if temp_skill and t == temp_skill[0]:
                temp_skill.pop(0)
            
            if t in temp_skill:
                count+=1
                break
    
    return len(skill_trees) - count