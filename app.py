import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
from streamlit.components.v1 import html

# 加载数据并训练模型
data = pd.read_csv('SMOTE5.8.csv')
y = data.iloc[:, 0]  # gold 在第一列
X = data.iloc[:, 1:]  # 其余为自变量

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
model.fit(X_train, y_train)

# 计算特征重要性
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': shap_sum})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# 选择前四个特征
top_features = importance_df.head(4)['feature'].tolist()

# Streamlit网页布局
st.set_page_config(layout="wide")  # 设置宽屏模式
st.sidebar.header("Patient characteristic")

# 初始化输入数据
input_data = {feature: X[feature].mean() for feature in X.columns}

# 在侧边栏中显示前四个特征供用户输入
for feature in top_features:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=float(X[feature].mean()))

# 将输入数据转换为数据框
input_df = pd.DataFrame([input_data])

# 预测和SHAP值计算
prediction = model.predict_proba(input_df)[0, 1]
shap_values_input = explainer.shap_values(input_df)

# 显示预测结果
st.title("EGV predict model")
st.subheader("EGV rate is: {:.2f}%".format(prediction * 100))

# 显示SHAP力图
st.subheader("SHAP Force explantation")
shap.initjs()

# 生成SHAP力图并保存为HTML文件
force_plot = shap.force_plot(explainer.expected_value, shap_values_input[0], input_df)
shap.save_html("force_plot.html", force_plot)

# 读取HTML文件内容
with open("force_plot.html", "r", encoding="utf-8") as file:
    force_plot_html = file.read()

# 使用Streamlit嵌入HTML内容
html(force_plot_html, height=400)

# 禁用 matplotlib 全局警告
st.set_option('deprecation.showPyplotGlobalUse', False)
