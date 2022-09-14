import re


"""
match() : 문자열의 처음부터 정규식과 매치되는지 조사한다.
search() : 문자열 전체를 검색하여 정슈식과 매치되는지 조사한다.
findall() : 정규식과 매치되는 모든 문자열을 리스트로 돌려준다.
finditer() : 정규식과 매치되는 모든 문자열을 반복 가능한 객체로 돌려준다.
"""

"""
[]
- [ ] 사이의 문자들 매치
- [a-zA-Z] : 알파벳 모두
- [0-9] : 숫자 모두
- [abc] : a,b,c 중 한개의 문자와 매치
"""
p = re.compile('[abc]+')
result = p.match("abcd")
print(result)