# 泰坦尼克号生存预测

本项目使用机器学习方法预测泰坦尼克号上乘客的生存情况。通过特征工程和随机森林模型，我们达到了约81%的准确率。

## 项目结构

```
titanic/
├── data/                    # 数据目录
│   ├── train.csv           # 训练数据
│   └── test.csv            # 测试数据
├── plots/                  # 分析图表
├── analyze.py              # 数据分析和可视化
├── main.py                 # 特征工程主程序
├── train_model.py          # 模型训练和评估
├── submission.csv          # 预测结果
└── README.md               # 项目说明
```

## 特征工程

我们创建了以下新特征：

1. **Title**：从姓名中提取头衔（Mr, Mrs, Miss等）
2. **FamilySize**：家庭成员总数（SibSp + Parch + 1）
3. **IsAlone**：是否独自一人
4. **FarePerPerson**：人均票价
5. **Age**：年龄（使用中位数填充缺失值）
6. **Deck**：从舱位号提取的甲板信息
7. **HasCabin**：是否有舱位信息

## 模型训练

### 使用的模型
- 随机森林分类器（Random Forest Classifier）
- 使用网格搜索进行超参数调优

### 最佳参数
```python
{
    'classifier__max_depth': 10,
    'classifier__min_samples_leaf': 1,
    'classifier__min_samples_split': 5,
    'classifier__n_estimators': 100
}
```

### 模型性能
- 验证集准确率：81.01%
- 精确率（Precision）：0.81
- 召回率（Recall）：0.81
- F1分数：0.81

## 特征重要性

根据模型分析，最重要的特征依次为：

1. FarePerPerson（人均票价）
2. Age（年龄）
3. Sex_female（女性）
4. Sex_male（男性）
5. Title_Mr（先生）
6. FamilySize（家庭规模）
7. Pclass_3（三等舱）
8. Title_Miss（小姐）
9. Pclass_1（一等舱）
10. Title_Mrs（夫人）
11. Title_Master（少爷）
12. Pclass_2（二等舱）
13. Title_Rare（稀有头衔）

## 如何运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行分析脚本（生成特征分析图表）：
   ```bash
   python analyze.py
   ```

3. 训练模型并生成预测：
   ```bash
   python train_model.py
   ```

4. 预测结果将保存在 `submission.csv` 文件中

## 后续优化方向

1. 尝试其他模型（如XGBoost、LightGBM等）
2. 进一步优化特征工程
   - 创建更多交互特征
   - 尝试不同的分箱策略
3. 调整模型超参数
4. 使用集成学习方法
5. 处理类别不平衡问题

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

[MIT](LICENSE)