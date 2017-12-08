# -*- coding: utf-8 -*-
# @Time       : 17-12-7 下午5:27
# @Author     : J.Y.Zhang
# @File       : PreFeatureEngineering.py
# @Description:

import pandas as pd
import numpy  as np
from collections import Counter

def dealNan(df):
    # 票价Nan setting silly values to nan
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    # 未知的船舱号
    df.Cabin = df.Cabin.fillna('Unknown')
    # 年龄平均填充
    meanAge = np.mean(df.Age)
    df.Age = df.Age.fillna(meanAge)
    # 登船口 众数填充
    word_counts = Counter(df.Embarked)
    mode = word_counts.most_common()
    modeEmbarked = mode[0][0]
    df.Embarked = df.Embarked.fillna(modeEmbarked)
    #modeEmbarked = mode(int(df.Embarked))
    #df.Embarked = df.Embarked.fillna(modeEmbarked)
    #    Fare per person

    return df

def extractDeck(df):
    # Turning cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    return df

def extractFamilysize(df):
    # Creating new family_size column
    df['Family_Size'] = df['SibSp'] + df['Parch']
    return df

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan

def extractTitle(df):
    # creating a title column from name
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    # replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1) #参数一是函数 参数二是函数应用的轴 0 是每一列 1 是每一行
    return df

def preFeatureEngineering(train,test):

    # nan
    train = dealNan(train)
    test = dealNan(test)

    # 名字里身份的特征
    train = extractTitle(train)
    test = extractTitle(test)


    # 家庭规模
    train = extractDeck(train)
    test = extractDeck(test)

    # 客舱位置
    train = extractFamilysize(train)
    test = extractFamilysize(test)

    data_type_dict = {'Pclass': 'ordinal', 'Sex': 'nominal',
                      'Age': 'numeric',
                      'Fare': 'numeric', 'Embarked': 'nominal', 'Title': 'nominal',
                      'Deck': 'nominal', 'Family_Size': 'ordinal'}

    return [train, test, data_type_dict]
def discretise_numeric(train, test, data_type_dict, no_bins= 10):
    N=len(train)
    M=len(test)
    test=test.rename(lambda x: x+N)
    joint_df=train.append(test)
    for column in data_type_dict:
        if data_type_dict[column]=='numeric':
            joint_df[column]=pd.qcut(joint_df[column], 10)
            data_type_dict[column]='ordinal'
    train=joint_df.ix[range(N)]
    test=joint_df.ix[range(N,N+M)]
    return train, test, data_type_dict

if __name__ == "__main__":
    trainpath = 'train.csv'
    testpath = 'test.csv'
    traindf = pd.read_csv(trainpath)
    testdf = pd.read_csv(testpath)
    #print(traindf.head(10))
    #
    traindf, testdf, data_type_dict = preFeatureEngineering(traindf,testdf)
    traindf, testdf, data_type_dict = discretise_numeric(traindf, testdf, data_type_dict)
    print(traindf.head(10))
    traindf.to_csv('traindf.csv', index=False)
    testdf.to_csv('testdf.csv', index= False)
    # create a submission file for kaggle
    predictiondf = pd.DataFrame(testdf['PassengerId'])
    predictiondf['Survived']=[0 for x in range(len(testdf))]
    predictiondf.to_csv('prediction.csv',header=None, index=False)

