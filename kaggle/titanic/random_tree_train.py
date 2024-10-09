import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 读取训练数据
train_df = pd.read_csv('train.csv')

# 预处理训练数据
# 假设年龄没有缺失值，如果有，可以用中位数填充
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# 将性别转换为数值型数据
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

# 将登船地点转换为数值型数据
label_encoder = LabelEncoder()
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'].astype(str))

# 特征和目标变量
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# 训练随机森林模型
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X, y)

# 保存模型
joblib.dump(random_forest_model, 'random_forest_model.pkl')

# 读取测试数据
test_df = pd.read_csv('test.csv')

# 预处理测试数据（与训练数据相同）
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Embarked'] = label_encoder.transform(test_df['Embarked'].astype(str))

# 特征
X_test = test_df[features]

# 使用模型进行预测
test_df['Survived'] = random_forest_model.predict(X_test)

# 保存结果到文件
test_df[['PassengerId', 'Survived']].to_csv('test_results.csv', index=False)
