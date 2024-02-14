"""
## 참고
- https://medium.com/@leejukyung/notion-py%EB%A1%9C-%EC%9E%90%EB%8F%99%EC%9C%BC%EB%A1%9C-%ED%8E%98%EC%9D%B4%EC%A7%80%EC%99%80-%ED%85%8C%EC%9D%B4%EB%B8%94-%EB%A7%8C%EB%93%A4%EA%B8%B0-5b685e8c8a5d
- https://velog.io/@bluestragglr/notion-py-Invalid-Input-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0
"""

from notion.client import NotionClient
from notion.block import PageBlock, ImageBlock
import requests
import os
import fitz


_MY_TOKEN = 'v02%3Auser_token_or_cookies%3A0Ph73_348Jje8KGveJWSyOhTjAtV1bQz9MTxdKRiVaxwU-frk1E2e8bFeT3TT4xm2L2hFi_zMEImQnXE-28JMrhQxAeF7iy4pSMc2lszPLGhL-GwRhMRPcK59zlM_M29K_cs'
_PAGE_URL = 'https://www.notion.so/Method-43b314778c9c4f84b324dc31a90774f9?pvs=4'
PDF_FILE_PATH = r""

if __name__ == '__main__':
    client = NotionClient(token_v2=_MY_TOKEN)
    page = client.get_block(_PAGE_URL)

    filename = os.path.split(PDF_FILE_PATH)[-1].split('.')[0]
    new_page = page.children.add_new(PageBlock) # 새로운 페이지 생성
    new_page.title = filename

    doc = fitz.open(PDF_FILE_PATH)

    for i, page in enumerate(doc):
        img = page.get_pixmap()
        img.save(os.path.join(r"C:\Users\cbigo\Desktop\temp", f"pdf_{i}.jpg"))
