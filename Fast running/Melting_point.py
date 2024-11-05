import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# 加载模型、填充器和标准化器
model = joblib.load("extra_trees_model.joblib")
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")

# 加载数据和 ChemBERTa 特征
X = np.load("X_data.npy")
y = np.load("y_data.npy")
smiles_features = np.load("smiles_features.npy")

# 合并数值特征和 ChemBERTa 特征（可选，用于确认数据结构一致性）
X = np.hstack([X, smiles_features])

# 十倍交叉验证设置
kf = KFold(n_splits=10, shuffle=True, random_state=42)


# 评估函数
def evaluate_model(model, X, y):
    r2_scores = []
    rmse_scores = []
    aard_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 模型训练与预测
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 计算 R2 分数
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)

        # 计算 AARD（平均绝对相对误差）
        aard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        aard_scores.append(aard)

    # 打印每折的结果
    print(f"Model: {model.__class__.__name__}")
    print(f"R2 Scores: {r2_scores}")
    print(f"Mean R2: {np.mean(r2_scores):.4f}")
    print("Best R^2 score across all folds: ", np.max(r2_scores))
    print(f"RMSE Scores: {rmse_scores}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"AARD Scores: {aard_scores}")
    print(f"Mean AARD: {np.mean(aard_scores):.4f}%\n")


# 评估加载的模型
evaluate_model(model, X, y)
