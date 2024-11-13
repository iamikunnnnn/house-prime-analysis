import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm

AllData = pd.read_csv(
    r'C:\Users\ikun\PycharmProjects\mypython\房价分析\house-prices-advanced-regression-techniques\train.csv')


# 猜想的几个特征

# 绘制散点图
Somedata = ["YearBuilt", "TotalBsmtSF", "GrLivArea"]
for i in Somedata:
    x = AllData[i]
    y = AllData['SalePrice']
    # 使用scatter函数绘制散点图
    plt.scatter(x, y)

    # 添加标题和轴标签
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel('SalePrice')
    # 显示图表
    plt.show()
# OverallQual箱型图

# 观察房价分布


# 绘制直方图

import pandas as pd
import matplotlib.pyplot as plt

# 创建一个示例Series
SalePrime = AllData["SalePrice"]

# 使用pandas的.hist()方法绘制直方图
SalePrime.hist(bins=100)  # bins参数指定直方图的柱形数量

# 显示图形
plt.show()

WordData = AllData[
    ["MSZoning", "Street", "LotShape", 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
     'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
     'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
     'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']]

NumData = AllData.drop(
    labels=["MSZoning", "Street", "LotShape", 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
            'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
            'SaleType', 'SaleCondition', "Alley", "PoolQC", "Fence", "MiscFeature"], axis=1, inplace=False)
print("非数值列为", WordData)
print("数值列为", NumData)


# 由于非数值列的类别都相对较少，故使用独热编码对对数值列进行处理
WordData_OH = pd.DataFrame()
encoder = OneHotEncoder(sparse_output=False)
for Col in WordData:
    # 对 Marital_Status列进行 OneHot 编码
    onehot_encoded = encoder.fit_transform(WordData[[Col]])
    # 获取 OneHot 编码后的列名
    try:
        feature_names = encoder.get_feature_names_out([Col])
    except AttributeError:
        feature_names = encoder.get_feature_names([Col])

    # 检查编码后的形状
    print("\nShape of OneHot Encoded Array:")
    print(onehot_encoded.shape)
    # 将 OneHot 编码后的数据转换为 DataFrame
    onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names)
    # 合并 OneHot 编码后的 DataFrame 和原始 DataFrame
    WordData_OH = pd.concat([WordData_OH, onehot_df], axis=1)
    # 打印 OneHot 编码后的 DataFrame
    print("\nOneHot Encoded DataFrame:")
    print(onehot_df)

    ## 独热编码可视化（各个类别的数量）   #由于导致图片过多暂时禁用
    #plt.figure(figsize=(10, 6))
    #onehot_df.sum().plot(kind='bar')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.title(Col)
    #plt.xticks(rotation=45)
    #plt.show()
print(WordData_OH)

# 相关系数矩阵分析数值数值相关性
corr_matrix = NumData.corr()

# 绘制相关系数矩阵的热力图
plt.figure(figsize=(10, 8))  # 设置图形大小

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # annot=True显示数值，cmap设置颜色映射，fmt设置数值格式

# 显示图形
plt.show()

# 卡方检验分析分类数据相关性（卡方检验适用于分类数据）
import numpy as np
from scipy.stats import chi2_contingency

for u in WordData.columns:
    for v in WordData.columns:
        contingency_table = pd.crosstab(WordData[u], WordData[v])

        # 进行卡方检验
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # 输出结果
        print(f"卡方统计量: {chi2}")
        print(f"P值: {p}")
        print(f"自由度: {dof}")
        print("期望频数:\n", expected)

        # 根据P值做出决策
        alpha = 0.05
        if p <= alpha:
            print("拒绝零假设，认为{}和{}之间存在显著的关联性。".format(u, v))
        else:
            print("不拒绝零假设，没有足够的证据表明{}和{}之间存在显著的关联性。".format(u, v))

# 找到的相关特征
FindData = NumData[
    ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
     'TotRmsAbvGrd', 'YearBuilt']]
corr_matrix_2 = FindData.corr()
print(corr_matrix_2)
# 绘制相关系数矩阵的热力图
plt.figure(figsize=(10, 8))  # 设置图形大小
sns.heatmap(corr_matrix_2, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5)  # annot=True显示数值，cmap设置颜色映射，fmt设置数值格式
# 显示图形
plt.show()
# ‘OverallQual’, ‘GrLivArea’ , ‘TotalBsmtSF’ 与 ‘SalePrice’有很强的相关性。

# ‘GarageCars’ 和 ‘GarageArea’ 也是相关性比较强的变量. 车库中存储的车的数量是由车库的面积决定的，它们就像双胞胎，所以不需要专门区分’GarageCars’ 和 ‘GarageArea’ ，所以我们只需要其中的一个变量。选择了’GarageCars’因为它与’SalePrice’ 的相关性更高一些。

# ‘TotalBsmtSF’ 和 ‘1stFloor’ and‘TotRmsAbvGrd’ 和 ‘GrLivArea’这两对也是双胞胎变量，选择 ‘TotalBsmtSF’ 和GrLivArea
# ‘YearBuilt’ 和 ‘SalePrice’相关性似乎不强。


# 双变量分析
FindData = FindData[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
plt.figure(figsize=(18, 15))
sns.pairplot(FindData)
# 显示图形
plt.show()

# 缺失值统计
total = AllData.isnull().sum().sort_values(ascending=False)
percent = (AllData.isnull().sum() / AllData.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# 当超过15%的数据都缺失的时候，删掉相关变量且假设该变量并不存在。
AllData_DLT = AllData.drop(AllData.columns[AllData.isnull().sum() > 1], axis=1)

AllData.replace(['NA'], np.nan, inplace=True)  # 识别空值


# 去空值
def fill_missing_with_interpolation(series):  # 使用拉格朗日插值法填补缺失值
    for i in range(len(series)):
        if pd.isnull(series.iloc[i]):
            series[i] = lagrange_interpolation(series, i)  # 对缺失值进行插值
    return series.astype(int)


def lagrange_interpolation(y, x):  # 拉格朗日插值函数

    k = 5  # 使用前后各k个数据点
    y = y[list(range(x - k, x)) + list(range(x + 1, x + 1 + k))]  # 获取插值所需数据点
    y = y[y.notnull()]  # 过滤掉缺失值
    return lagrange(y.index, list(y))(x)  # 计算插值


features = AllData_DLT[['SalePrice', 'GrLivArea', 'TotalBsmtSF']]
for feature in features:
    AllData_DLT[feature] = fill_missing_with_interpolation(AllData_DLT[feature])  # 使用插值法填补列的缺失值

# 异常值处理

# 方案一：
# 假设data是你的数据集

# mean = np.mean(AllData_DLT['GrLivArea'])
# std = np.std(AllData_DLT['GrLivArea'])
# median = np.median(AllData_DLT['GrLivArea'])
# AllData_DLT['GrLivArea'].loc[np.abs(AllData_DLT['GrLivArea'] - median) > 3 * np.std(AllData_DLT['GrLivArea'])] = median


# 方案二：
x = AllData_DLT['GrLivArea']
y = AllData_DLT['SalePrice']
plt.scatter(x, y)
for i, txt in enumerate(x):
    plt.annotate(f'({txt:.2f}, {y[i]:.2f})', (x[i], y[i]))
# 添加标题和轴标签
plt.title('price')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
# 显示图表
plt.show()

# 显示结果

abnormal_values = [4676.00, 5642.00]
AllData_DLT = AllData_DLT[~AllData_DLT['GrLivArea'].isin(abnormal_values)]
print(AllData_DLT)
x = AllData_DLT['GrLivArea']
y = AllData_DLT['SalePrice']
plt.scatter(x, y)

# 添加标题和轴标签
plt.title('price')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
# 显示图表
plt.show()

# 回归模型建立准备：--------------------------------------------------------------------------
# TotalBsmtSF含0值，无法对数变换，取出非零值再变换
AllData_DLT['HasBsmt'] = 0
AllData_DLT.loc[AllData_DLT['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
AllData_DLT = AllData_DLT[AllData_DLT['HasBsmt'] != 0]

features = AllData_DLT[['SalePrice', 'GrLivArea', 'TotalBsmtSF']]

for i in features:
    # 拟合正态分布
    print("{i}的正态性检验")
    sns.distplot(AllData_DLT[i], fit=norm)
    fig = plt.figure()
    res = stats.probplot(AllData_DLT[i], plot=plt)

    # Q-Q图验证正态性（Q-Q图需要满足数据属于同分布）
    # 使用SciPy的stats模块绘制Q-Q图
    stats.probplot(AllData_DLT[i], dist="norm", plot=plt)
    plt.show()
    # 不符合正态性

    # 进行对数变换
    AllData_DLT[i] = np.log(AllData_DLT[i])
    print("{i}对数化后的正态性检验")
    sns.distplot(AllData_DLT[i], fit=norm)
    fig = plt.figure()
    res = stats.probplot(AllData_DLT[i], plot=plt)

    # Q-Q图验证正态性（Q-Q图需要满足数据属于同分布）
    # 使用SciPy的stats模块绘制Q-Q图
    stats.probplot(AllData_DLT[i], dist="norm", plot=plt)
    plt.show()

    # 结果符合正态性

# 检验同方差性
plt.scatter(AllData_DLT[AllData_DLT['TotalBsmtSF'] > 0]['TotalBsmtSF'],
            AllData_DLT[AllData_DLT['TotalBsmtSF'] > 0]['SalePrice']);
plt.show()


plt.scatter(AllData_DLT['GrLivArea'], AllData_DLT['SalePrice'])
plt.show()


print(AllData_DLT)
