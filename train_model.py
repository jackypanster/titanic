import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from main import prepare_data

def prepare_features():
    # 准备数据
    train_df, test_df = prepare_data('train.csv', 'test.csv')
    
    # 选择特征
    features = ['Pclass', 'Sex', 'Title', 'FarePerPerson', 'Age', 'FamilySize']
    X = train_df[features]
    y = train_df['Survived']
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, test_df, features

def build_model():
    # 定义预处理步骤
    numeric_features = ['FarePerPerson', 'Age', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Title']
    
    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 创建模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return model

def train_and_evaluate():
    # 准备数据
    X_train, X_val, y_train, y_val, test_df, features = prepare_features()
    
    # 构建模型
    model = build_model()
    
    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("开始训练模型...")
    grid_search.fit(X_train, y_train)
    
    # 输出最佳参数
    print("\n最佳参数:", grid_search.best_params_)
    
    # 在验证集上评估
    y_val_pred = grid_search.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\n验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_val_pred))
    
    # 在整个训练集上重新训练
    print("\n在整个训练集上重新训练...")
    final_model = grid_search.best_estimator_
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    # 在测试集上预测
    X_test = test_df[features]
    test_predictions = final_model.predict(X_test)
    
    # 保存结果
    output = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    output.to_csv('submission_optimized.csv', index=False)
    print("\n预测结果已保存到 submission_optimized.csv")
    
    # 特征重要性
    try:
        # 获取预处理后的特征名称
        preprocessor = final_model.named_steps['preprocessor']
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        
        # 获取特征名称
        numeric_features = ['FarePerPerson', 'Age', 'FamilySize']
        categorical_features = ['Pclass', 'Sex', 'Title']
        
        # 获取独热编码后的特征名称
        ohe_feature_names = ohe.get_feature_names_out(categorical_features)
        
        # 合并所有特征名称
        feature_names = numeric_features + list(ohe_feature_names)
        
        # 获取特征重要性
        importances = final_model.named_steps['classifier'].feature_importances_
        
        # 创建DataFrame并排序
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性:")
        print(feature_importance)
        
    except Exception as e:
        print("\n无法获取特征重要性:", e)
        raise e  # 重新抛出异常以便调试

if __name__ == "__main__":
    train_and_evaluate()