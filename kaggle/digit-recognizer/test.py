# predict.py
import torch
import pandas as pd
import cnn_model  # 导入自定义的CNN模型

# 初始化模型并加载权重
model = cnn_model.CNN()
model.load_state_dict(torch.load('cnn_model100.pth'))
model.eval()

# 读取测试数据
test_data = pd.read_csv('test.csv')
test_features = test_data.values.reshape(-1, 1, 28, 28).astype('float32')

# 进行预测
predictions = []
with torch.no_grad():
    for i in range(test_features.shape[0]):
        image = torch.tensor(test_features[i]).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predictions.append(predicted.item())

# 生成预测文件
submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv('submission100.csv', index=False)
