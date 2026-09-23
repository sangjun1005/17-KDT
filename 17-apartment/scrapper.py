import requests
import xml.etree.ElementTree as ET

import os
from dotenv import load_dotenv

load_dotenv()

SERVICEKEY = os.getenv('SERVICEKEY')

def fetch_apt_trade(LAWD_CD, DEAL_YMD):

    # LAWD_CD='27230'
    # DEAL_YMD='202606'
    serviceKey=SERVICEKEY

    URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"

    params={
        'LAWD_CD':LAWD_CD,
        'DEAL_YMD':DEAL_YMD,
        'serviceKey':serviceKey
    }

    response = requests.get(URL, params=params)
    # print(response.content)

    root = ET.fromstring(response.content)
    items = root.findall('./body/items/item')
    results = []

    for item in items:
        row = {}
        for child in item:
            row[child.tag] = child.text
        
        results.append(row)

    return results