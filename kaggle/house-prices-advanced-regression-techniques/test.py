import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder

# 加载训练好的随机森林模型
rf = load('random_forest_regressor.joblib')

# 加载之前训练时使用的OneHotEncoder
encoder = load('one_hot_encoder.joblib')  # 确保您已经保存了encoder

# 读取测试数据
test_data = pd.read_csv('test.csv')

# 确定哪些列是类别型的（非数值型的）
categorical_cols = test_data.select_dtypes(include=['object']).columns

# 如果有类别型数据，使用OneHotEncoder进行编码
if len(categorical_cols) > 0:
    # 转换数据
    encoded_data = encoder.transform(test_data[categorical_cols])
    # 将稀疏矩阵转换为密集数组
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
    # 删除原始的类别型列
    test_data = test_data.drop(categorical_cols, axis=1)
    # 将编码后的数据合并到测试数据集
    test_data = pd.concat([test_data, encoded_df], axis=1)

# 保留 'Id' 列用于创建提交文件
test_ids = test_data['Id']

# 删除 'Id' 列，因为它不是模型输入的一部分
test_data = test_data.drop(['Id'], axis=1)

# 使用模型进行预测
test_predictions = rf.predict(test_data)

# 创建提交文件
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions})

# 保存提交文件为 CSV，不包含索引列
submission.to_csv('my_submission.csv', index=False)
