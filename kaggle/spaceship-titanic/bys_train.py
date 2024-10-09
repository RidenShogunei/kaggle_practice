import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import joblib

# 读取训练数据
train_df = pd.read_csv('train.csv')

# 预处理数据
label_encoders = {}
for column in ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']:
    le = LabelEncoder()
    train_df[column] = le.fit_transform(train_df[column])
    label_encoders[column] = le

# 填充缺失值
for column in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    train_df[column].fillna(train_df[column].median(), inplace=True)

# 特征和目标变量
features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X = train_df[features]
y = train_df['Transported']

# 训练朴素贝叶斯模型
nb_model = GaussianNB()
nb_model.fit(X, y)

# 保存模型
joblib.dump(nb_model, 'nb_model.pkl')

# 读取测试数据
test_df = pd.read_csv('test.csv')

# 预处理测试数据
for column, le in label_encoders.items():
    # 只转换在训练数据集中见过的标签
    test_df[column] = test_df[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# 填充测试数据中的缺失值
for column in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    test_df[column].fillna(test_df[column].median(), inplace=True)

# 特征
X_test = test_df[features]

# 使用模型进行预测
test_df['Transported'] = nb_model.predict(X_test)

# 保存结果到文件
submission_df = test_df[['PassengerId', 'Transported']]
submission_df.to_csv('submission.csv', index=False)
