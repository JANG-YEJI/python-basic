#!/usr/bin/env python
# coding: utf-8

# 1. https://finance.naver.com/item/sise_day.nhn?code=005930&page=1
# 2. 종목 10개, page 5개
# 3. 시세 Dataframe 만들기
#     - 종목코드, 날짜, 종가, 전일대비, 시가, 고가, 저가, 거래양


import pandas as pd
import requests



headers = {"user-agent":"Mozilla/5.0"}
df = pd.DataFrame()
code = ["005930", "051910", "035720", "063160", "185750",
        "006400", "006980", "096770", "068290", "028300"]
for i in range(len(code)):
    for j in range(1, 6):
        url = "https://finance.naver.com/item/sise_day.nhn?code=" + code[i] + "&page=" + str(j)

        resp = requests.get(url, headers = headers)
        resp.text

        pd_read = pd.read_html(resp.text)
        new_df = pd_read[0]
        new_df = new_df.dropna(axis=0)
        new_df.insert(0, "종목코드", code[i])
        df = pd.concat([df, new_df])



df
