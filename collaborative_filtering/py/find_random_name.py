#appleAppData에서 랜덤으로 앱 이름 50개 추출해서 저장했습니다

import pandas as pd
import numpy as np
import random

np.random.seed(23)

file_path = '../csv/preprocessed_appleAppData.csv'
df = pd.read_csv(file_path)

num_name = 50
names = random.sample(df['App_Name'].tolist(), num_name)

output_file = names

output_path = '../csv/app_name.txt'
with open(output_path, 'w', encoding='utf-8') as file:
    for name in names:
        file.write(f"{name}\n")

print (output_path,'에 저장되었습니다')