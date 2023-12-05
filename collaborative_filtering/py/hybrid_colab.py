import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from sklearn.metrics import mean_squared_error
from math import sqrt

# col로 사용할 앱 이름들
app_name_path = '../csv/app_name.txt'
with open(app_name_path, 'r', encoding='utf-8') as file:
    app_name = [line.strip() for line in file]

# 평가한 csv 읽어오기
rating_data = pd.read_csv('../csv/ratings_data.csv')

# 빈 값 0으로 대체
rating_data = rating_data.fillna(0)

# id, item, rating으로 변환
rating_data = rating_data.melt(id_vars=['id'], value_vars=app_name, var_name='item', value_name='rating')

# 평가 점수는 1~5
reader = Reader(rating_scale=(1, 5))

# df 형식을 Dataset 형식으로 바꾸기
data = Dataset.load_from_df(rating_data[['id', 'item', 'rating']], reader)

# 사용자 기반 협업 모델 학습
user_model = KNNBasic(sim_options={'user_based': True})

# 아이템 기반 협업 모델 학습
item_model = KNNBasic(sim_options={'user_based': False})

# 실험을 위한 가중치 범위 설정 (0%부터 100%까지 5%씩 증가)
user_weight_range = range(0, 101, 5)

best_rmse_user = float('inf')
best_user_model = None
best_user_weight = 0

# 사용자 기반 모델의 가중치를 변화시키면서 평가
for user_weight in user_weight_range:
    # 가중치를 퍼센트로 변환
    user_weight /= 100.0
    item_weight = 1 - user_weight

    # 협업 필터링 가중 평균 모델 학습 (실제 사용할 때는 여기서 데이터를 분리하여 학습하면 됨)
    cv_results_user = cross_validate(user_model, data, measures=['RMSE'], cv=5, verbose=False)

    # 전체 K-fold의 평균 RMSE 계산
    avg_rmse_user = sum(cv_results_user['test_rmse']) / len(cv_results_user['test_rmse'])

    # 현재 가중치 조합이 더 나은 경우 업데이트
    if avg_rmse_user < best_rmse_user:
        best_rmse_user = avg_rmse_user
        best_user_model = user_model
        best_user_weight = user_weight

# 최적의 사용자 기반 모델과 가중치 출력
print(f'최적의 사용자 기반 모델의 가중치: {best_user_weight * 100}%')
print(f'최적의 사용자 기반 모델의 평균 RMSE: {best_rmse_user}')
