def make_code_list(m_list):
    code_list = list()
    for m in m_list:
        
        if m.isalpha():
            code_list.append(m)
        
        else:
            code_list[-1] = code_list[-1]+"#"
            
    return code_list

def time_diff(start_time, end_time):
    start_minute, start_second = start_time.split(":")
    end_minute, end_second = end_time.split(":")

    end = int(end_minute)*60 + int(end_second)
    start = int(start_minute)*60 + int(start_second)
    
    return end-start
    
def search(template, source):
    
    for i in range(len(template) - len(source)+1):
        if template[i:i+len(source)] == source:
            return True
        
    return False
    
def solution(m_list, musicinfos):
    
    m_list = make_code_list(m_list)
    result_title = ""
    max_time = 0
    
    for musicinfo in musicinfos:
        start_time, end_time, title, codes = musicinfo.split(",")
        
        during_time = time_diff(start_time, end_time)
        codes = make_code_list(codes)
        if len(codes) < during_time:
            a = during_time // len(codes)
            b = during_time % len(codes)
            codes = codes * a + codes[:b]
        else:
            codes = codes[:during_time]
        
        if search(codes, m_list):
            if max_time < during_time:
                max_time = during_time
                result_title = title

    return "(None)" if result_title == "" else result_title