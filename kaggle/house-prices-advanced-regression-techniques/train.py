import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

# 加载数据
data = pd.read_csv('train.csv')
data = data.drop("Id", axis=1)
print(f"data{data}")
# 确定哪些列是类别型的（非数值型的）
categorical_cols = data.select_dtypes(include=['object']).columns

# 如果有类别型数据，使用OneHotEncoder进行编码
if len(categorical_cols) > 0:
    # 初始化 OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    # 拟合数据
    encoder.fit(data[categorical_cols])
    # 转换数据
    encoded_data = encoder.transform(data[categorical_cols])
    # 将稀疏矩阵转换为密集数组
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
    # 删除原始的类别型列
    data = data.drop(categorical_cols, axis=1)
    # 将编码后的数据合并到原始数据集
    data = pd.concat([data, encoded_df], axis=1)
    # 保存 OneHotEncoder
    dump(encoder, 'one_hot_encoder.joblib')
print(f"data after process{data}")
# 读取特征和结果
# 确保这里不包含目标列和不需要的列
X = data.drop(['SalePrice'], axis=1)
y = data['SalePrice']
print(f"x{X}")
print(f"y{y}")
# 划分训练数据
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模块
rf = RandomForestRegressor(n_estimators=800, random_state=42)

# 训练
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_val)

# 计算损失并打印
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")

# 保存模型
dump(rf, 'random_forest_regressor.joblib')
