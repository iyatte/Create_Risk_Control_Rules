# -*- coding: utf-8 -*-
"""
@Time    : 2019/12/4 15:09
@Author  : yany
@File    : example.py
"""
import pandas as pd
from tree_model.Tree import DataPreProcessing
from tree_model.Tree import Model
from tree_model.Tree import Visualization
from tree_model.Tree import FeatureImportance
from tree_model.Tree import GetRule
from tree_model.Tree import OOT


if __name__ == "__main__":
    # ###########################   参数配置   ###########################
    # ########################### 较常更改参数 ###########################
    # =========== 模型参数设置 ===========
    # 选取树模型，可换为'cart','rf','lgb','xgb','id3'
    tree_type = 'lgb'
    # 数据切分比，不需要提前切分，[0, 1)之间的一个数值，若为0，则无测试集，所有数据用于训练
    test_size = 0.2
    # 标签列
    y_columns = 'overdueday_flag_7'
    # y_columns = 'overdue_day_7_flag'
    # 敏感词(模糊匹配的方式删除x_columns的数据)
    ignore_word = ['overdueday_flag_7', 'usermobile', 'overdue_day_7_flag']
    # 是否需要继续OOT,如果需要设置为True,否则设置为False
    is_oot = False
    # =========== 数据路径设置 ===========
    # 训练数据路径
    train_data_path = r'./raw_data/final_data_test.csv'
    # 时间外验数据路径
    checking_data_path = r'./raw_data/final_data_test.csv'
    # =========== 保存路径配置 ===========
    # 保存总路径(最后请加/）
    sav_path = './result/'
    # =========== 模型参数设置 ===========
    # 树深
    max_depth = 5
    # 学习率，控制每次迭代更新权重时的步长，值越小，训练越慢。
    learning_rate = 0.01
    # 主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别，可取'balanced' or None
    class_weight = 'balanced'
    # 树的个数
    n_estimators = 10
    # x_columns在下面，可自行配置，如果不自行配制的话，就是删除y_columns和能模糊匹配到的ignore_word后的剩余字段,
    # 如自行配置，请在下方更改x_column 并 置ignore_word为[]

    # =========== 保存路径配置 ===========
    # 保存总路径
    # 树模型图片保存路径
    picture_path = sav_path+tree_type+'/'+'plot/'
    # 模型保存路径
    model_path = sav_path+tree_type+'/'+'model/'
    # 指标重要性
    importance_path = sav_path+tree_type+'/'+'importance/'
    # 规则保存路径
    rule_path = sav_path+tree_type+'/'+'rule/'
    # oot结果保存路径
    oot_path = sav_path+tree_type+'/'+'oot/'
    # =========== 加载数据 ===========
    df = pd.read_csv(train_data_path)
    # 确定x_columns
    x_columns = df.columns.tolist()
    x_columns.remove(y_columns)
    # ########################### 数据预处理 ###########################
    # 初始化数据预处理模块
    data_pre = DataPreProcessing()
    x_columns = data_pre.ignore_func(x_columns, ignore_word)
    # 缺失值处理,放入为df，具体参数详情点入查看
    # null_function 第四个参数为0，则为不进行填充，缺失值将变为np.nan
    new_data, x_columns_new = data_pre.null_function(df, x_columns, 1, 1, ignore_columns=['overdue_day_7_flag'], is_delete=0.8,
                                                     null_value=[-999, '-999', '\\N', 'NULL'])
    # 依据iv值进行数据筛选
    # iv_threshold为iv的取值区间，也可用columns_num,取IV值排名前columns_num个(从大到小)
    # 返回，第一个为list,为选取columns结果，第二个为df，为所有字典对应的iv值
    x_columns_final, df_iv = data_pre.select_columns(new_data, x_columns_new, y_columns, iv_threshold=[0.04, 0.5],
                                                     columns_num=None)

    # ########################### 建模 ###########################
    # 初始化建模模块
    model_ = Model()
    # 模型预测
    model_final = model_.train_model(tree_type, new_data, x_columns_final, y_columns, max_depth, learning_rate,
                                     n_estimators, class_weight, test_size)
    # 模型保存
    model_.save_model(model_final, model_path)

    # 模型加载
    model_path = model_path+'model.pkl'
    new_model = model_.load_model(model_path)

    # ########################### 可视化 ###########################
    # 初始化可视化模块
    # class_names = list(set(new_data[y_columns].astype('str')))
    # visual_ = Visualization()
    # visual_.plot(new_model, tree_type, picture_path, x_columns_final, class_names)

    # ########################### 指标重要性 ###########################
    # 初始化指标重要性模块
    importance_ = FeatureImportance()
    importance_.importance(new_model, x_columns_final, importance_path)

    # ########################### 规则提取 ###########################
    # 初始化规则提取模块
    # conclude_max参数可设置提起规则的层级，比如之前设置树深为10，可提取前N条，为None是全部提取
    # new_model 模型
    # rule_path 规则提取后存储路径，rule_path必须为xlsx文件
    conclude_max = None
    rule_ = GetRule()
    rule_.get_rule(tree_type, new_model, new_data, x_columns_final, y_columns, conclude_max, rule_path)

    # ########################### 时间外验 ###########################
    # oot筛选的条件共有8个，将依据设置的条件对condition进行自动筛选，将留下的规则送入oot进行测评，
    # 如需自己筛选，则删选完后更改rule_path即可，select_condition置为[]
    # condition_select中若为1 < x < 2情况，请分为两个条件，即['x > 1', 'x > 2']  注：符号前后请留一个空格，一个！！空格！！
    # ########################### 时间外验 ###########################
    # 初始化时间外验模块
    # new_data 时间外验数据，可自定义
    # x_columns_final 特征名称
    # y_columns 目标列名称
    # rule_path_new xlsx的sheet名必须为bad和good
    # oot_path 时间外验结果保存路径
    # es_sample_rate,es_sample_num,es_real_bad,es_real_good,es_real_bad_rate,es_real_good_rate,es_auc,es_recall
    rule_path_new = rule_path + 'rule.xlsx'
    condition_select = ['es_real_bad_rate > 0', 'es_real_good_rate > 0']
    if is_oot:
        oot_ = OOT()
        oot_.oot(checking_data_path, x_columns_final, y_columns, rule_path_new, oot_path, condition_select)


