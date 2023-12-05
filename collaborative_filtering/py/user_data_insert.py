#유저 1,000명, 앱 50개, 빈칸 비율 45%의 유저-아이템 테이블을 만들었습니다

import pandas as pd
import numpy as np


np.random.seed(23)
app_name_path = '../csv/app_name.txt'
with open(app_name_path, 'r', encoding='utf-8') as file:
    app_name = [line.strip() for line in file]

app_name.insert(0,'id')

# 행, 열의 개수
num_rows = 1000  # 유저 1000명
num_columns = len(app_name) - 1 # 어플 개수 (id는 빼고 계산)

# 빈칸 비율
empty_percentage = 0.45

# 각 평가 점수는 1~5의 랜덤 정수
data = np.random.choice([1, 2, 3, 4, 5], size=(num_rows, num_columns), p=[0.2, 0.2, 0.2, 0.2, 0.2])

# 랜덤하게 선택한 45%는 빈칸으로 바꿉니다
mask = np.random.rand(*data.shape) < empty_percentage
# 정수는 np.nan이 안돼서 형태 변환했습니다
data = data.astype(float)
data[mask] = np.nan

# id로 쓸 column을 추가했습니다.
data = np.insert(data, 0, values=np.arange(1, num_rows + 1), axis=1)

# dataframe으로 변환합니다
df = pd.DataFrame(data, columns=app_name)

# csv로 변환했습니다
df.to_csv('../csv/ratings_data.csv', index=False)

print("CSV file 'ratings_data.csv' has been created.")
