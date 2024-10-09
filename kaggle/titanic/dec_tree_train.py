import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 读取CSV文件
df = pd.read_csv('train.csv')

# 数据预处理
# 填充年龄缺失值
df['Age'].fillna(df['Age'].median(), inplace=True)

# 转换性别为数值型
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 转换登船地点为数值型
label_encoder = LabelEncoder()
df['Embarked'] = label_encoder.fit_transform(df['Embarked'].astype(str))

# 选择特征和目标变量
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
target = df['Survived']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 创建决策树模型
clf = DecisionTreeClassifier(random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
predictions = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')


