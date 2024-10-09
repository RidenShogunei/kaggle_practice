import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 读取训练数据
train_df = pd.read_csv('train.csv')

# 预处理训练数据
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
label_encoder = LabelEncoder()
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'].fillna('S'))  # 假设缺失的Embarked用'S'填充

# 特征和目标变量
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# 训练SVM模型
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # 使用线性核
svm_model.fit(X, y)

# 保存模型
joblib.dump(svm_model, 'svm_model.pkl')

# 读取测试数据
test_df = pd.read_csv('test.csv')

# 预处理测试数据（与训练数据相同）
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)  # 假设缺失的Fare用中位数填充
test_df['Embarked'] = label_encoder.transform(test_df['Embarked'].fillna('S'))  # 假设缺失的Embarked用'S'填充

# 特征
X_test = test_df[features]

# 使用模型进行预测
test_df['Survived'] = svm_model.predict(X_test)

# 保存结果到文件
test_df[['PassengerId', 'Survived']].to_csv('test_results_svm.csv', index=False)
