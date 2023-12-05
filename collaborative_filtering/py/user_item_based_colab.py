from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
from surprise import dump
import pandas as pd

# col로 사용할 앱 이름들
app_name_path = '../csv/app_name.txt'
with open(app_name_path, 'r', encoding='utf-8') as file:
    app_name = [line.strip() for line in file]

# 평가한 csv 읽어오기
rating_data = pd.read_csv('../csv/ratings_data.csv')

# 빈 값 0으로 대체
rating_data = rating_data.fillna(0)

# id, item, rating으로 변환합니다. item 열은 앱 이름들을 묶는 역할
rating_data = rating_data.melt(id_vars=['id'], value_vars=app_name, var_name='item', value_name='rating')

# 평가 점수는 1~5
reader = Reader(rating_scale=(1, 5))

# df 형식을 Dataset 형식으로 바꾸기
data = Dataset.load_from_df(rating_data[['id', 'item', 'rating']], reader)

# k-fold에서 k가 바뀔 때의 rmse 값을 저장할 리스트
mean_rmse_list = []

# k가 1에서 4까지
for k_value in range(1, 6):
    # KNN 알고리즘을 사용하여 사용자 기반 협업 필터링 모델을 생성합니다
    
    # user based 일때
    model = KNNBasic(k=k_value, sim_options={'user_based': True})

    # item based 일때
    # model = KNNBasic(k=k_value, sim_options={'user_based': False})

    # 교차 검증 수행
    results = cross_validate(model, data, measures=['RMSE'], cv=5, verbose=False)

    # 모든 폴드의 RMSE 평균을 계산
    mean_rmse = results['test_rmse'].mean()

    # 리스트에 추가
    mean_rmse_list.append(mean_rmse)

    # 모델을 저장하고 RMSE를 출력
    #user based 일때
    model_name = f'../model/user_model_k{k_value}.pkl'

    #item based 일때
    # model_name = f'../model/item_model_k{k_value}.pkl'
    dump.dump(model_name, algo=model)

# 리스트에 저장된 RMSE 값을 출력
for k_value, mean_rmse in enumerate(mean_rmse_list, start=1):

    # user based 일때
    print(f'User-Based-Colob-Model with k={k_value} saved with RMSE: {mean_rmse}')

    #item based 일때
    # print(f'Item-Based-Colob-Model with k={k_value} saved with RMSE: {mean_rmse}')