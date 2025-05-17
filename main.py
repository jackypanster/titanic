import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss", 
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Rare",
            "Rev": "Rare",
            "Col": "Rare",
            "Major": "Rare",
            "Mlle": "Miss",
            "Countess": "Rare",
            "Ms": "Miss",
            "Lady": "Rare",
            "Jonkheer": "Rare",
            "Don": "Rare",
            "Dona": "Rare",
            "Mme": "Mrs",
            "Capt": "Rare",
            "Sir": "Rare"
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # 创建数据副本
        df = X.copy()
        
        # 1. 提取Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].map(self.title_mapping)
        df['Title'] = df['Title'].fillna('Rare')  # 将未映射的标题标记为Rare
        
        # 2. 创建家庭相关特征
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        
        # 3. 处理Cabin特征
        df['Deck'] = df['Cabin'].str[0]  # 提取甲板信息
        df['Deck'] = df['Deck'].fillna('U')  # 填充缺失值为'U' (Unknown)
        df['HasCabin'] = df['Cabin'].notnull().astype(int)  # 是否有舱位信息
        
        # 4. 处理Age特征
        # 先填充中位数，后续可以使用更复杂的方法
        age_medians = df.groupby(['Title', 'Pclass'])['Age'].median()
        df['Age'] = df.apply(
            lambda row: age_medians.get((row['Title'], row['Pclass']), df['Age'].median()) 
            if pd.isnull(row['Age']) else row['Age'], 
            axis=1
        )
        
        # 5. Age分箱
        df['AgeBin'] = pd.cut(
            df['Age'].astype(int), 
            bins=[0, 12, 20, 40, 120], 
            labels=['Child', 'Teenager', 'Adult', 'Elderly']
        )
        
        # 6. 处理Fare特征
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # 填充测试集中的缺失值
        df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False) + 1  # 修改为使用数字标签
        
        # 7. 处理Embarked
        df['Embarked'] = df['Embarked'].fillna('S')  # 使用众数填充
        
        # 8. 创建票价每人特征
        df['FarePerPerson'] = df['Fare'] / (df['FamilySize'] + 1e-8)  # 避免除以0
        
        # 删除不再需要的列
        df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        
        return df

def prepare_data(train_path, test_path):
    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 添加Survived列到测试集（设置为-1作为占位符）
    test_df['Survived'] = -1
    
    # 合并数据集进行特征工程
    combined = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    
    # 应用特征工程
    fe = FeatureEngineering()
    combined_featured = fe.transform(combined)
    
    # 分割回训练集和测试集
    train_featured = combined_featured[combined_featured['Survived'] != -1].copy()
    test_featured = combined_featured[combined_featured['Survived'] == -1].copy()
    test_featured = test_featured.drop('Survived', axis=1)
    
    return train_featured, test_featured

if __name__ == "__main__":
    # 测试特征工程
    train_df, test_df = prepare_data('train.csv', 'test.csv')
    print("训练集形状:", train_df.shape)
    print("测试集形状:", test_df.shape)
    print("\n训练集列名:", train_df.columns.tolist())
    print("\n测试集列名:", test_df.columns.tolist())
    print("\n训练集前5行:")
    print(train_df.head())