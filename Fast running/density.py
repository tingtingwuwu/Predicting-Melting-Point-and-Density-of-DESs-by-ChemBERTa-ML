import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 加载模型、填充器、标准化器
model = joblib.load("xgb_regressor_model.joblib")
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")

# 加载处理好的数据
X_filtered_imputed = np.load("X_filtered_imputed.npy")
y_filtered = np.load("y_filtered.npy")

# 加载 ChemBERTa 特征（可选）
smiles1_features = np.load("smiles1_features.npy")
smiles2_features = np.load("smiles2_features.npy")
smiles_features = np.hstack([smiles1_features, smiles2_features])


# 定义 AARD 计算函数
def mean_absolute_relative_deviation(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 进行十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmses, aards, r2s = [], [], []

for train_index, test_index in kf.split(X_filtered_imputed):
    X_train, X_test = X_filtered_imputed[train_index], X_filtered_imputed[test_index]
    y_train, y_test = y_filtered[train_index], y_filtered[test_index]

    # 标准化
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用加载的模型进行预测
    y_pred = model.predict(X_test_scaled)

    # 计算评价指标
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    aard = mean_absolute_relative_deviation(y_test, y_pred)

    rmses.append(rmse)
    r2s.append(r2)
    aards.append(aard)

# 输出结果
print("Cross-validation RMSE for each fold: ", rmses)
print("Average RMSE across all folds: ", np.mean(rmses))
print("Cross-validation AARD for each fold: ", aards)
print("Average AARD across all folds: ", np.mean(aards))
print("Cross-validation R^2 for each fold: ", r2s)
print("Best R^2 score across all folds: ", np.max(r2s))
