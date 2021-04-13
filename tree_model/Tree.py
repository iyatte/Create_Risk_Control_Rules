# -*- coding: utf-8 -*-
"""
@Time    : 2019/12/4 10:45
@Author  : yany
@File    : Tree.py
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from joblib import load, dump
import pydot
import graphviz
import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DataPreProcessing(object):
    def __init__(self):
        pass

    @staticmethod
    def ignore_func(x_columns, ignore_word):
        if len(ignore_word) > 0:
            for ig_ in ignore_word:
                for x_word in x_columns:
                    if ig_ in x_word:
                        x_columns.remove(x_word)
        return x_columns

    @staticmethod
    def null_function(df, x_columns, is_change, is_fill, ignore_columns, is_delete=None, null_value=None):
        """
        :param df: 需要缺失值处理的df
        :param x_columns: 特征名称
        :param is_change: 缺失值处理后，是否需要改变列的类型, 1 为改变，0为不变
        :param is_fill: 会把存在的缺失值处理成np.nan，0 为不填充， 1为根据类别填充为'-999'或-999，其他情况请田0后自定义
        :param ignore_columns: list 需要忽略的列
        :param is_delete: None 不删除，若为[0-1]之间的值，则为缺失率大于该值则删除
        :param null_value: list 里面存在数据中可能存在np.nan的值，比如[-999,'-999','\\N']
        :return: df
        """
        # 将null_value中的值替换为np.nan
        for i in range(len(x_columns)):
            if x_columns[i] not in ignore_columns:
                df[x_columns[i]] = df[x_columns[i]].apply(lambda x: np.nan if x in null_value else x)
        if is_delete is not None:
            for i in range(len(x_columns)):
                if x_columns[i] not in ignore_columns:
                    if df[x_columns[i]].isna().sum() / len(df) > is_delete:
                        del df[x_columns[i]]
        # 有些因存在\N等字段，导致本应为number的字段变为str，缺失值填充也从'-999'
        if is_change == 1:
            for i in range(len(x_columns)):
                if x_columns[i] not in ignore_columns:
                    try:
                        df[x_columns[i]] = df[x_columns[i]].astype('float')
                    except Exception as e:
                        print(e)
                    continue
        if is_fill != 0:
            for i in range(len(x_columns)):
                if x_columns[i] not in ignore_columns and x_columns[i] in df.columns.tolist():
                    if df[x_columns[i]].dtype == 'object':
                        df.loc[df[x_columns[i]].isna(), x_columns[i]] = '-999'
                    else:
                        df.loc[df[x_columns[i]].isna(), x_columns[i]] = -999
        x_columns_new = [i for i in x_columns if i in df.columns.tolist()]
        return df, x_columns_new

    @staticmethod
    def calc_iv(x_var, y_var):
        n_0 = np.sum(x_var == 0)
        n_1 = np.sum(y_var == 1)
        n_0_group = np.zeros(np.unique(x_var).shape)
        n_1_group = np.zeros(np.unique(x_var).shape)
        for i in range(len(np.unique(x_var))):
            n_0_group[i] = y_var[(x_var == np.unique(x_var)[i]) & (y_var == 0)].count()
            n_1_group[i] = y_var[(x_var == np.unique(x_var)[i]) & (y_var == 1)].count()
        iv = np.sum((n_0_group / n_0 - n_1_group / n_1) * np.log((n_0_group / n_0) / (n_1_group / n_1)))
        return iv

    def select_columns(self, df, x_columns, y_columns, iv_threshold, columns_num=None):
        df_bak = df.copy()
        for i_ in range(len(x_columns)):
            if len(df_bak[x_columns[i_]].drop_duplicates()) > 10:
                df_bak[x_columns[i_]] = pd.qcut(df_bak[x_columns[i_]], 10, duplicates='drop',
                                                retbins=False).astype('str')
        iv_result = []
        for x_index in range(len(x_columns)):
            print('*'*5, x_index, '*'*5, len(x_columns))
            iv_result.append(self.calc_iv(df_bak[x_columns[x_index]], df_bak[y_columns]))
        df_iv = pd.DataFrame(x_columns, columns=['columns_'])
        df_iv['iv_value'] = iv_result
        df_iv = df_iv.sort_values(by='iv_value', ascending=False).reset_index(drop=True)
        if iv_threshold is not None:
            iv_threshold.sort()
            columns_result = df_iv[(df_iv.iv_value >= iv_threshold[0]) & (df_iv.iv_value <= iv_threshold[1])]\
                .columns_.tolist()
            if len(columns_result) == 0:
                raise Exception('no feature selected')
        else:
            columns_result = df_iv.loc[:columns_num-1, 'columns_'].tolist()

        return columns_result, df_iv


class Model(object):
    def __init__(self):
        pass

    @staticmethod
    def test_model(y_test, pre_y, pre_y_pro):
        # 混淆矩阵
        # 核心评估指标
        accuracy_s = accuracy_score(y_test, pre_y)  # 准确率
        recall_s = recall_score(y_test, pre_y)  # 召回率
        auc_s = roc_auc_score(y_test, pre_y_pro)  # auc
        print('AUC:', auc_s, '准确率:', accuracy_s, '召回率:', recall_s)

    @staticmethod
    def data_split(df, x_columns, y_columns, test_size):
        if test_size >= 1:
            raise Exception('test_size must be [0, 1)')
        elif test_size == 0:
            return df[x_columns], df[y_columns]
        else:
            x_train, x_test, y_train, y_test = train_test_split(df[x_columns], df[y_columns], test_size=test_size,
                                                                stratify=df[y_columns])
            return x_train, x_test, y_train, y_test

    def ori_model(self, model, test_size, df, x_columns, y_columns):
        if test_size == 0:
            x_train, y_train = self.data_split(df, x_columns, y_columns, test_size)
            model.fit(x_train, y_train)
        else:
            x_train, x_test, y_train, y_test = self.data_split(df, x_columns, y_columns, test_size)
            model.fit(x_train, y_train)
            pre_y = model.predict(x_test)
            pre_y_pro = model.predict_proba(x_test)
            self.test_model(y_test, pre_y, pre_y_pro[:, 1])
        return model

    def rf_model(self, df, x_columns, y_columns, max_depth, learning_rate, n_estimators, class_weight, test_size,
                 criterion):
        rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, class_weight=class_weight)
        rf = self.ori_model(rf, test_size, df, x_columns, y_columns)
        return rf

    def decision_model(self, df, x_columns, y_columns, max_depth, learning_rate, n_estimators, class_weight, test_size,
                       criterion):
        decision_tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, class_weight=class_weight)
        decision_tree = self.ori_model(decision_tree, test_size, df, x_columns, y_columns)
        return decision_tree

    def lgb_model(self, df, x_columns, y_columns, max_depth, learning_rate, n_estimators, class_weight, test_size,
                  criterion):
        lgb_model = lgb.LGBMClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                                       class_weight=class_weight)
        lgb_model = self.ori_model(lgb_model, test_size, df, x_columns, y_columns)
        return lgb_model

    def xgb_model(self, df, x_columns, y_columns, max_depth, learning_rate, n_estimators, class_weight, test_size,
                  criterion):
        xgb_model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                                      class_weight=class_weight)
        xgb_model = self.ori_model(xgb_model, test_size, df, x_columns, y_columns)
        return xgb_model

    def train_model(self, tree_type, df, x_columns, y_columns, max_depth, learning_rate, n_estimators, class_weight,
                    test_size):
        model_dict = {'xgb': self.xgb_model,
                      'lgb': self.lgb_model,
                      'rf': self.rf_model,
                      'cart': self.decision_model,
                      'id3': self.decision_model}
        criterion_dict = {'xgb': '1',
                          'lgb': '1',
                          'rf': '1',
                          'cart': 'gini',
                          'id3': 'entropy'}
        model = model_dict[tree_type](df, x_columns, y_columns, max_depth, learning_rate, n_estimators, class_weight,
                                      test_size, criterion_dict[tree_type])
        return model

    @staticmethod
    def save_model(model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        dump(model, path+'model.pkl')

    @staticmethod
    def load_model(path):
        model = load(path)
        return model


class Visualization(object):
    def __init__(self):
        pass

    @staticmethod
    def tree_plot(tree, sav_path, is_decision_tree, class_names, x_columns):
        """
        :param tree: 树模型
        :param sav_path: 图片保存地址
        :param is_decision_tree: 是否决策树，1为决策树 2为随机森林
        :param class_names: list 类别名称，按排序
        :param x_columns: list 变量名,为送入模型的x特征的顺序
        :return: 无返回，请到图片保存路径查看
        """
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        if is_decision_tree != 1:
            tree_num = len(tree.estimators_)
            for i in range(tree_num):

                export_graphviz(tree.estimators_[i], out_file="./tree_model/cong/tree.dot", class_names=class_names,
                                feature_names=x_columns, impurity=False, filled=True)
                # 展示可视化图
                (graph,) = pydot.graph_from_dot_file('./tree_model/cong/tree.dot')
                graph.write_jpg(sav_path + 'rf_tree' + str(i) + '.jpg')
                graph.write_pdf(sav_path + 'rf_pdf_tree' + str(i) + '.pdf')
        else:
            export_graphviz(tree, out_file="./tree_model/cong/tree.dot", class_names=class_names,
                            feature_names=x_columns, impurity=False, filled=True)
            # 展示可视化图
            (graph,) = pydot.graph_from_dot_file('./tree_model/cong/tree.dot')
            graph.write_jpg(sav_path + 'tree.jpg')
            graph.write_pdf(sav_path + 'tree_pdf.pdf')

    @staticmethod
    def lgb_plot(lgb_model, sav_path, is_decision_tree, class_names, x_columns):
        """
        :param lgb_model: 模型
        :param sav_path: 图片保存地址
        :param is_decision_tree: 是否决策树，1为决策树 2为随机森林
        :param class_names: list 类别名称，按排序
        :param x_columns: 特征名称
        :return: 无返回，请到图片保存路径查看
        """
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        b = lgb_model.booster_.dump_model()
        tree_num = len(b['tree_info'])
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        for i in range(tree_num):
            lgb.plot_tree(lgb_model, tree_index=i, figsize=(20, 8), show_info=['split_gain'])
            plt.savefig(sav_path + 'lgb_tree' + str(i) + '.png', dpi=1000)
            plt.savefig(sav_path + 'lgb_pdf_tree' + str(i) + '.pdf', dpi=1000)

    @staticmethod
    def create_feature_map(features):
        outfile = open('./tree_model/cong/xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()

    def xgb_plot(self, xgb_model, sav_path, is_decision_tree, class_names, x_columns):
        """
        :param xgb_model: 模型
        :param sav_path: 图片保存地址
        :param is_decision_tree: 是否决策树
        :param class_names: 类别名称
        :param x_columns: list 变量名,为送入模型的x特征的顺序
        :return: 无返回，请到图片保存路径查看
        """
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        self.create_feature_map(x_columns)
        tree_num = xgb_model.n_estimators
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        for i in range(tree_num):
            xgb.plot_tree(xgb_model, num_trees=i, fmap='./tree_model/cong/xgb.fmap')
            plt.savefig(sav_path + 'xgb_tree' + str(i) + '.jpg', dpi=1000)
            plt.savefig(sav_path + 'xgb_pdf_tree' + str(i) + '.pdf', dpi=1000)
        print('end !')

    def plot(self, tree_model, tree_type, sav_path, x_columns, class_names):
        model_dict = {'xgb': self.xgb_plot,
                      'lgb': self.lgb_plot,
                      'rf': self.tree_plot,
                      'cart': self.tree_plot,
                      'id3': self.tree_plot}
        tree_type_dict = {'xgb': 0, 'lgb': 0, 'rf': 0, 'cart': 1, 'id3': 1}
        plot_model = model_dict[tree_type]
        plot_model(tree_model, sav_path, tree_type_dict[tree_type], class_names, x_columns)


class FeatureImportance(object):
    def __init__(self):
        pass

    @staticmethod
    def importance(model, x_columns, importance_path):
        importance_ = model.feature_importances_
        df = pd.DataFrame(x_columns)
        df['importance_'] = importance_
        df = df.sort_values(by='importance_', ascending=False)
        if not os.path.exists(importance_path):
            os.makedirs(importance_path)
        df.to_csv(importance_path+'importance.csv', index=False)


class GetRule(object):
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def tree_rule(self, model_, x_columns, is_decision_tree, tree_index=None):
        """
        :param model_: 模型
        :param x_columns: 特征数据的字段名，注意，与输入数据的顺序保持一致
        :param is_decision_tree: 是否决策树模型，如果是，为1，如果是森林，为0
        :param tree_index: 树的index
        :return:
        """
        if is_decision_tree == 1:
            decision_tree = model_
            r = export_text(decision_tree, feature_names=x_columns)
            r = r.replace('|', '')
            r = r.replace('---', '')
            b = r.split('\n')
            result_all = []
            dic_ = {}
            for c_values in range(len(b)):
                if 'class:' in b[c_values]:
                    result_ = list(dic_.values())
                    result_.insert(0, b[c_values].replace('   ', '').replace(' class: ', ''))
                    if tree_index is not None:
                        result_.insert(0, tree_index)
                    result_all.append(result_)
                elif 'truncated' in b[c_values]:
                    pass
                else:
                    cnt_ = b[c_values].count('   ')
                    dic_[cnt_] = b[c_values].replace('   ', '').replace('  ', ' ')[1:]
        else:
            result_all = []
            for i in range(len(model_.estimators_)):
                rule_bak = self.tree_rule(model_.estimators_[i], x_columns, 1, i)
                result_all.extend(rule_bak)
        return result_all

    def lgb_list_all_dict(self, tree_1, dict_, path, x_columns):
        sign_dic = {'>=': '<',
                    '>': '<=',
                    '<=': '>',
                    '<': '>='}
        if isinstance(tree_1, dict):  # 使用isinstance检测数据类型
            if 'leaf_value' not in tree_1.keys():
                temp_value = tree_1['left_child']
                path_ = path+'left_'
                dict_[path_] = str(x_columns[tree_1['split_feature']]) + ' ' + str(tree_1['decision_type']) + ' ' \
                    + str(tree_1['threshold'])
                dict_ = self.lgb_list_all_dict(temp_value, dict_, path_, x_columns)
                temp_value = tree_1['right_child']
                path_ = path+'righ_'
                dict_[path_] = str(x_columns[tree_1['split_feature']]) + ' ' + str(sign_dic[tree_1['decision_type']]) \
                    + ' ' + str(tree_1['threshold'])
                dict_ = self.lgb_list_all_dict(temp_value, dict_, path_, x_columns)
                # print("%s : %s" % (temp_key, temp_value))
                # 自我调用实现无限遍历
            else:
                dict_[path+'flag_'] = tree_1['leaf_value']
        return dict_

    def lgb_dic_to_rule_bak(self, tree_dict, tree_index):
        df_1 = pd.DataFrame.from_dict(tree_dict, orient='index').reset_index()
        df_1.columns = ['columns_', 'sign']
        df_1['rank_'] = df_1.columns_.apply(lambda x: len(x.split('_')) - 1)
        df_1['flag_'] = df_1.columns_.apply(lambda x: 1 if 'flag' in x else 0)
        df_2 = df_1[df_1.flag_ == 1].reset_index(drop=True)

        result_all = []
        for i in range(len(df_2)):
            dict_ = {}
            cnt_s = 1
            df_3 = df_2.iloc[[i]]
            dict_['score'] = self.sigmoid(df_3['sign'].values[0])
            while df_3.rank_.values[0] != 1:
                df_3 = df_1[(df_1.columns_ == df_3.columns_.values[0][:-5]) & (df_1.rank_ == df_3.rank_.values[0]-1)]
                dict_[cnt_s] = df_3['sign'].values[0]
                cnt_s += 1
            result_ = list(dict_.values())
            result_.insert(0, tree_index)
            result_all.append(result_)
        return result_all

    # 模型提取规则
    def lgb_rule(self, lgb_model, x_columns, is_decision_tree, tree_index=None):
        a = lgb_model.booster_
        b = a.dump_model()
        tree_num = len(b['tree_info'])
        result_all = []
        for i in range(tree_num):
            tree_1 = b['tree_info'][i]['tree_structure']
            path = ''
            dict_ = {}
            tree_dict = self.lgb_list_all_dict(tree_1, dict_, path, x_columns)
            result_all.extend(self.lgb_dic_to_rule_bak(tree_dict, i))
        return result_all

    def xgb_dic_to_rule_bak(self, tree_dict, tree_index):
        df_1 = pd.DataFrame.from_dict(tree_dict, orient='index').reset_index()
        df_1.columns = ['columns_', 'sign']
        df_1.sign = df_1.sign.apply(lambda x: float(x.replace('leaf=', '')) if 'leaf=' in x else x)
        df_1['rank_'] = df_1.columns_.apply(lambda x: len(x.split('_')) - 1)
        df_1['flag_'] = df_1.columns_.apply(lambda x: 1 if 'flag' in x else 0)
        df_2 = df_1[df_1.flag_ == 1].reset_index(drop=True)
        result_all = []
        for i in range(len(df_2)):
            dict_ = {}
            cnt_s = 1
            df_3 = df_2.iloc[[i]]
            dict_['score'] = self.sigmoid(float(df_3['sign'].values[0]))
            while df_3.rank_.values[0] != 1:
                df_3 = df_1[(df_1.columns_ == df_3.columns_.values[0][:-5]) & (df_1.rank_ == df_3.rank_.values[0]-1)]
                dict_[cnt_s] = df_3['sign'].values[0]
                cnt_s += 1
            result_ = list(dict_.values())
            result_.insert(0, tree_index)
            result_all.append(result_)
        return result_all

    def xgb_test(self, b_dict, b_sign, cnt_new, dict_all, path):
        if 'leaf' not in b_dict[cnt_new]:
            path_ = path+'yess_'
            temp_value = b_sign[float(b_dict[cnt_new].split(' ')[1].split(',')[0].replace('yes=', ''))]
            dict_all[path_] = temp_value
            cnt_new_ = float(b_dict[cnt_new].split(' ')[1].split(',')[0].replace('yes=', ''))
            dict_all = self.xgb_test(b_dict, b_sign, cnt_new_, dict_all, path_)
            path_ = path+'nooo_'
            temp_value = b_sign[float(b_dict[cnt_new].split(' ')[1].split(',')[1].replace('no=', ''))]
            dict_all[path_] = temp_value
            cnt_new_ = float(b_dict[cnt_new].split(' ')[1].split(',')[1].replace('no=', ''))
            dict_all = self.xgb_test(b_dict, b_sign, cnt_new_, dict_all, path_)
        else:
            dict_all[path + 'flag_'] = b_dict[cnt_new]
        return dict_all

    @staticmethod
    def create_feature_map(features):
        outfile = open('./tree_model/cong/xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()

    def xgb_rule(self, xgb_model, x_columns, is_decision_tree, tree_index=None):
        self.create_feature_map(x_columns)
        xgb_model.get_booster().dump_model('./tree_model/cong/xgb_model.txt',  fmap='./tree_model/cong/xgb.fmap')
        with open('./tree_model/cong/xgb_model.txt', 'r') as f:
            txt_model = f.read()
        txt_model = txt_model.split('booster')
        result_all = []
        for i in range(1, len(txt_model)):
            if 'yes' in txt_model[i]:
                b = txt_model[i].replace('\t', '').split('\n')
                b_dict = {}
                for b_values in b[1:-1]:
                    b_dict[float(b_values.split(':')[0])] = b_values.split(':')[1]
                b_sign = {}
                for b_sign_value in b_dict.values():
                    if 'leaf' not in b_sign_value:
                        b_sign[float(b_sign_value.split(' ')[1].split(',')[0].replace('yes=', ''))] = \
                            b_sign_value.split(' ')[0].replace('[', '').replace(']', '').replace('<', ' < ')
                        b_sign[float(b_sign_value.split(' ')[1].split(',')[1].replace('no=', ''))] = \
                            b_sign_value.split(' ')[0].replace('[', '').replace(']', '').replace('<', ' >= ')
                cnt_new = 0
                path = ''
                dict_all = {}
                tree_dict = self.xgb_test(b_dict, b_sign, cnt_new, dict_all, path)
                result_bak = self.xgb_dic_to_rule_bak(tree_dict, len(txt_model)-i-1)
                result_all.extend(result_bak)
        return result_all

    @staticmethod
    def good_rate(a, begin_index, x_test, y_test):
        ori_len = len(x_test)
        x_test.loc[:, 'y'] = y_test
        x_test_bak = x_test.copy()
        label_ = a.label_
        a = a.to_list()
        for i in range(begin_index, len(a)):
            if pd.notna(a[i]):
                func_ = a[i].split(' ')
                if func_[1] == '>=':
                    x_test = x_test[x_test[func_[0]] >= float(func_[2])]
                elif func_[1] == '<=':
                    x_test = x_test[x_test[func_[0]] <= float(func_[2])]
                elif func_[1] == '>':
                    x_test = x_test[x_test[func_[0]] > float(func_[2])]
                elif func_[1] == '<':
                    x_test = x_test[x_test[func_[0]] < float(func_[2])]
        good_rat = round((len(x_test) / ori_len) * 100, 2)
        good_num = len(x_test)
        real_bad = len(x_test[x_test.y == 1])
        real_good = len(x_test[x_test.y == 0])
        if good_num > 0:
            real_bad_rate = np.round(real_bad / good_num * 100, 2)
            real_good_rate = np.round(real_good / good_num * 100, 2)
        else:
            real_bad_rate = None
            real_good_rate = None
        # auc
        if label_ > 0.5:
            x_test_bak['y_pre'] = 0
            x_test_bak.loc[x_test.index.to_list(), 'y_pre'] = 1
        else:
            x_test_bak['y_pre'] = 1
            x_test_bak.loc[x_test.index.to_list(), 'y_pre'] = 0
        auc = roc_auc_score(x_test_bak.y, x_test_bak.y_pre)
        recall = recall_score(x_test_bak.y, x_test_bak.y_pre)
        return pd.Series([good_rat, good_num, real_bad, real_good, real_bad_rate, real_good_rate, auc, recall],
                         index=['es_sample_rate', 'es_sample_num', 'es_real_bad', 'es_real_good', 'es_real_bad_rate',
                                'es_real_good_rate', 'es_auc', 'es_recall'])

    def get_rule(self, tree_type, model_, df, x_columns, y_columns, conclude_max, save_path):
        rule_dict = {'xgb': self.xgb_rule,
                     'lgb': self.lgb_rule,
                     'rf': self.tree_rule,
                     'cart': self.tree_rule,
                     'id3': self.tree_rule}
        rule_func = rule_dict[tree_type]
        # model_, x_columns, is_decision_tree, tree_index=None
        tree_type_dict = {'xgb': 0, 'lgb': 0, 'rf': 0, 'cart': 1, 'id3': 1}
        if tree_type in ['cart', 'id3']:
            tree_index = 0
        else:
            tree_index = None
        rule_ = rule_func(model_, x_columns, tree_type_dict[tree_type], tree_index)
        df_rule = pd.DataFrame(rule_)
        rule_columns = df_rule.columns.tolist()
        rule_columns[0] = 'tree_index'
        rule_columns[1] = 'label_'
        for i in range(2, len(rule_columns)):
            rule_columns[i] = 'condition_' + str(i - 2)
        df_rule.columns = rule_columns
        # ###############统计字段出现次数
        rule_columns = [i for i in df_rule.columns if 'condition' in i]
        count_ = []
        for i in range(len(rule_columns)):
            count_.extend(df_rule[df_rule[rule_columns[i]].notna()][rule_columns[i]].apply(lambda x: x.split(' ')[0])
                          .tolist())
        df_count = pd.DataFrame(count_, columns=['columns'])
        df_count = df_count.groupby('columns', as_index=False)['columns'].agg({'count': 'count'}).\
            sort_values(by='count', ascending=False)
        # ###############截取condition层次
        if conclude_max is not None:
            df_rule = df_rule.iloc[:, :conclude_max+2]
        es_feature = ['es_sample_rate', 'es_sample_num', 'es_real_bad', 'es_real_good', 'es_real_bad_rate',
                      'es_real_good_rate', 'es_auc', 'es_recall']
        for i in range(len(es_feature)):
            df_rule.insert(i+2, es_feature[i], np.ones(len(df_rule)))
        df_rule.label_ = df_rule.label_.astype('float')
        df_bad = df_rule[df_rule.label_ >= 0.5].reset_index(drop=True)
        df_good = df_rule[df_rule.label_ < 0.5].reset_index(drop=True)
        begin_index = 2+len(es_feature)
        ori_bad_rate = np.round(len(df[df[y_columns] == 1]) / len(df) * 100, 2)
        if len(df_bad) > 0:
            df_bad[es_feature]\
                = df_bad.apply(lambda x: self.good_rate(x, begin_index, df[x_columns], df[y_columns]), axis=1)
            df_bad.insert(2, '原数据坏人占比', [ori_bad_rate]*len(df_bad))
        if len(df_good) > 0:
            df_good[es_feature]\
                = df_good.apply(lambda x: self.good_rate(x, begin_index, df[x_columns], df[y_columns]), axis=1)
            df_good.insert(2, '原数据坏人占比', [ori_bad_rate]*len(df_good))

        df_info = pd.read_csv('./tree_model/cong/columns_info.csv')
        df_info = df_info.drop_duplicates(subset='columns')
        df_count = df_count.merge(df_info, how='left', on='columns')
        df_count = df_count.fillna("tree_model.cong.columns_info.csv 无该字段，请进行补充")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        writer = pd.ExcelWriter(save_path+'rule'+'.xlsx')
        df_bad.to_excel(writer, sheet_name='bad', index=False)
        df_good.to_excel(writer, sheet_name='good', index=False)
        df_count.to_excel(writer, sheet_name='columns_count', index=False)
        writer.save()


class OOT(object):
    def __init__(self):
        pass

    @staticmethod
    def good_rate(a, begin_index, x_test, y_test):
        ori_len = len(x_test)
        x_test.loc[:, 'y'] = y_test
        x_test_bak = x_test.copy()
        label_ = a.label_
        a = a.to_list()
        for i in range(begin_index, len(a)):
            if pd.notna(a[i]):
                func_ = a[i].split(' ')
                if func_[1] == '>=':
                    x_test = x_test[x_test[func_[0]] >= float(func_[2])]
                elif func_[1] == '<=':
                    x_test = x_test[x_test[func_[0]] <= float(func_[2])]
                elif func_[1] == '>':
                    x_test = x_test[x_test[func_[0]] > float(func_[2])]
                elif func_[1] == '<':
                    x_test = x_test[x_test[func_[0]] < float(func_[2])]
        good_rat = round((len(x_test) / ori_len) * 100, 2)
        good_num = len(x_test)
        real_bad = len(x_test[x_test.y == 1])
        real_good = len(x_test[x_test.y == 0])
        if good_num > 0:
            real_bad_rate = np.round(real_bad / good_num * 100, 2)
            real_good_rate = np.round(real_good / good_num * 100, 2)
        else:
            real_bad_rate = None
            real_good_rate = None
        # auc
        if label_ > 0.5:
            x_test_bak['y_pre'] = 0
            x_test_bak.loc[x_test.index.to_list(), 'y_pre'] = 1
        else:
            x_test_bak['y_pre'] = 1
            x_test_bak.loc[x_test.index.to_list(), 'y_pre'] = 0
        auc = roc_auc_score(x_test_bak.y, x_test_bak.y_pre)
        recall = recall_score(x_test_bak.y, x_test_bak.y_pre)
        return pd.Series([good_rat, good_num, real_bad, real_good, real_bad_rate, real_good_rate, auc, recall],
                         index=['es_sample_rate', 'es_sample_num', 'es_real_bad', 'es_real_good', 'es_real_bad_rate',
                                'es_real_good_rate', 'es_auc', 'es_recall'])

    @staticmethod
    def is_hit(a, x_test, begin_index):
        a = a.to_list()
        result_ = []
        n = 0
        for i in range(begin_index, len(a)):
            if pd.notna(a[i]):
                n += 1
                func_ = a[i].split(' ')
                if func_[1] == '>=' and x_test[func_[0]] >= float(func_[2]):
                    result_.append(1)
                elif func_[1] == '<=' and x_test[func_[0]] <= float(func_[2]):
                    result_.append(1)
                elif func_[1] == '>' and x_test[func_[0]] > float(func_[2]):
                    result_.append(1)
                elif func_[1] == '<' and x_test[func_[0]] < float(func_[2]):
                    result_.append(1)
        if n == np.sum(result_):
            return 1
        else:
            return 0

    @staticmethod
    def con_select(df_rule, condition_select):
        for i in range(len(condition_select)):
            if pd.notna(condition_select[i]):
                func_ = condition_select[i].split(' ')
                if func_[1] == '>=':
                    df_rule = df_rule[df_rule[func_[0]] >= float(func_[2])]
                elif func_[1] == '<=':
                    df_rule = df_rule[df_rule[func_[0]] <= float(func_[2])]
                elif func_[1] == '>':
                    df_rule = df_rule[df_rule[func_[0]] > float(func_[2])]
                elif func_[1] == '<':
                    df_rule = df_rule[df_rule[func_[0]] < float(func_[2])]
        return df_rule

    def ott_cal(self, df_rule, condition_select, df, x_columns, y_columns, save_path, flag_):
        df_rule = self.con_select(df_rule, condition_select)
        rule_columns = [i for i in df_rule.columns if 'condition' in i]
        df_rule_new = df_rule[rule_columns]
        df_rule_new.insert(0, 'rule_flag', ['rule_'+flag_+'_'+str(i) for i in range(0, len(df_rule))])
        df_rule_new.insert(1, 'label_', df_rule.label_)
        df_rule_new_2 = df_rule_new[['rule_flag']].copy()
        es_feature = ['es_sample_rate', 'es_sample_num', 'es_real_bad', 'es_real_good', 'es_real_bad_rate',
                      'es_real_good_rate', 'es_auc', 'es_recall']
        for i in range(len(es_feature)):
            df_rule_new_2.insert(i+1, es_feature[i], np.ones(len(df_rule_new_2)))
        begin_index = 2
        a = df_rule_new.apply(lambda x: self.good_rate(x, begin_index, df[x_columns], df[y_columns]), axis=1)
        df_rule_new_2[es_feature] = a
        columns_new = [i+'_ori' for i in es_feature]
        df_rule_new_2[columns_new] = df_rule[es_feature]
        df_rule_new_2['坏人率'] = len(df[df['y_columns'] == 1]) / len(df)

        df_rule_new_3 = pd.DataFrame([str(i+1) for i in range(len(df))], columns=['data_index'])
        for rule_ in range(len(df_rule_new)):
            rule_new = df_rule_new.iloc[rule_]
            df_rule_new_3[rule_new[0]] = df.apply(lambda x: self.is_hit(rule_new, x, begin_index), axis=1)
        df_rule_new_3.insert(1, 'true_class', df[y_columns])
        df_rule_new_3.insert(2, 'hit_count', df_rule_new_3.iloc[:, 2:].sum(axis=1))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        writer = pd.ExcelWriter(save_path+'oot_'+flag_+'_rule.xlsx')
        df_rule_new.to_excel(writer, sheet_name='rule_ori', index=False)
        df_rule_new_2.to_excel(writer, sheet_name='oot_result_1', index=False)
        df_rule_new_3.to_excel(writer, sheet_name='oot_result_2', index=False)
        writer.save()

    def oot(self, checking_data_path, x_columns, y_columns, rule_path, save_path, condition_select):
        df = pd.read_csv(checking_data_path)
        df_1 = pd.read_excel(rule_path, sheet_name='bad')
        df_2 = pd.read_excel(rule_path, sheet_name='good')
        if len(df_1) > 0:
            self.ott_cal(df_1, condition_select, df, x_columns, y_columns, save_path, 'bad')
        if len(df_2) > 0:
            self.ott_cal(df_2, condition_select, df, x_columns, y_columns, save_path, 'good')
