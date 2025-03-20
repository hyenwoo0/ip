import requests
import json
import pandas as pd

# API 설정
API_URL = "http://apis.data.go.kr/1160100/service/GetStocDiviInfoService/getDiviInfo"
SERVICE_KEY = "FJu4pK8qDPaxr5Wly0tMcZl7pvXx3+cNAO3wmnyeONIjDv+b2frtzJ8vK950T0zyzXKy+SRASjsoZT8kRBjFKQ=="  # API 키 입력

# API 요청 파라미터
params = {
    "serviceKey": SERVICE_KEY,
    "numOfRows": 100,  # 적절한 데이터 크기 조정
    "pageNo": 1,
    "resultType": "json",
}

# API 요청
response = requests.get(API_URL, params=params)

#  응답 상태 확인
if response.status_code == 200:
    try:
        data = response.json()
    except json.JSONDecodeError:
        print(" JSON 디코딩 오류 발생! 응답 내용을 확인하세요.")
        print(" Response Content:", response.text)
        exit()
else:
    print(f" API 요청 실패! HTTP 상태 코드: {response.status_code}")
    print(" Response Content:", response.text)
    exit()

#  데이터 추출
items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

#  중요 데이터 필터링
if items:
    df = pd.DataFrame(items)

    # 필요한 컬럼만 선택
    columns_to_keep = [
        "basDt", "stckIssuCmpyNm", "dvdnBasDt", "cashDvdnPayDt", "stckDvdnRcdNm",
        "stckGenrDvdnAmt", "stckGenrCashDvdnRt", "stckGenrDvdnRt"
    ]
    df_filtered = df[columns_to_keep]

    #  컬럼명을 한글로 변경
    columns_mapping = {
        "basDt": "기준일",
        "stckIssuCmpyNm": "발행회사명",
        "dvdnBasDt": "배당기준일",
        "cashDvdnPayDt": "현금배당지급일",
        "stckDvdnRcdNm": "배당유형",
        "stckGenrDvdnAmt": "배당금액",
        "stckGenrCashDvdnRt": "현금배당률(%)",
        "stckGenrDvdnRt": "주식배당률(%)"
    }
    df_filtered = df_filtered.rename(columns=columns_mapping)

    # CSV 저장
    file_path = "filtered_stock_dividend_data.csv"
    df_filtered.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f" CSV 파일 저장 완료: {file_path}")

    #  표 형태로 출력
    print("\n 필터링된 데이터 (최대 10개 미리보기) ")
    print(df_filtered.head(10))  # 상위 10개 행만 출력
else:
    print("⚠️ 데이터가 없습니다.")
