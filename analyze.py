import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import prepare_data

# 设置绘图样式
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def plot_categorical_feature(df, feature, target='Survived'):
    """绘制分类特征与生存率的关系"""
    plt.figure(figsize=(12, 5))
    
    # 计数图
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=feature, hue=target)
    plt.title(f'Count of {feature} by {target}')
    plt.xticks(rotation=45)
    
    # 生存率
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x=feature, y=target)
    plt.title(f'Survival Rate by {feature}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    return plt

def plot_numerical_feature(df, feature, target='Survived'):
    """绘制数值特征与生存率的关系"""
    plt.figure(figsize=(12, 4))
    
    # 分布图
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=feature, hue=target, element='step', stat='density', common_norm=False)
    plt.title(f'Distribution of {feature} by {target}')
    
    # 箱线图
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=target, y=feature)
    plt.title(f'{feature} by {target}')
    
    plt.tight_layout()
    return plt

def analyze_features(save_plots=True, show_plots=False):
    # 创建保存图表的目录
    if save_plots and not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 准备数据
    train_df, _ = prepare_data('train.csv', 'test.csv')
    
    # 分类特征分析
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'AgeBin', 'FareBin', 'IsAlone', 'HasCabin']
    for feature in categorical_features:
        if feature in train_df.columns:
            plt = plot_categorical_feature(train_df, feature)
            if save_plots:
                plt.savefig(f'plots/{feature}_analysis.png')
            if show_plots:
                plt.show()
            plt.close()
    
    # 数值特征分析
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'FarePerPerson']
    for feature in numerical_features:
        if feature in train_df.columns:
            plt = plot_numerical_feature(train_df, feature)
            if save_plots:
                plt.savefig(f'plots/{feature}_analysis.png')
            if show_plots:
                plt.show()
            plt.close()
    
    # 特征相关性热力图
    plt.figure(figsize=(12, 10))
    # 选择数值型列
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    # 计算相关性
    corr = train_df[numeric_cols].corr()
    # 绘制热力图
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    if save_plots:
        plt.savefig('plots/feature_correlation.png')
    if show_plots:
        plt.show()
    plt.close()

if __name__ == "__main__":
    import os
    analyze_features(save_plots=True, show_plots=False)