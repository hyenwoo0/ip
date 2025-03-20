import requests
import pandas as pd

# 🔹 API 설정
API_URL = "http://apis.data.go.kr/1160100/service/GetStocDiviInfoService/getDiviInfo"
SERVICE_KEY = "FJu4pK8qDPaxr5Wly0tMcZl7pvXx3+cNAO3wmnyeONIjDv+b2frtzJ8vK950T0zyzXKy+SRASjsoZT8kRBjFKQ=="  # 🔑 API 키 입력

# 🔹 API 요청 파라미터
params = {
    "serviceKey": SERVICE_KEY,
    "numOfRows": 10000,  # 테스트용으로 10개만 가져오기
    "pageNo": 1,
    "resultType": "json",
}

# 🔹 API 요청
response = requests.get(API_URL, params=params)

# 🔹 응답 확인 및 예외 처리
try:
    data = response.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

    if items:
        df = pd.DataFrame(items)
        df.to_csv("stock_dividend_data.csv", index=False, encoding="utf-8-sig")
        print("✅ CSV 파일 저장 완료: stock_dividend_data.csv")
    else:
        print("⚠️ 데이터가 없습니다.")

except requests.exceptions.JSONDecodeError:
    print("❌ JSON 디코딩 오류 발생! 응답 내용을 확인하세요.")
    print("🔹 Response Status Code:", response.status_code)
    print("🔹 Response Content:", response.text)
