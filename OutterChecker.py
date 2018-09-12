# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:34:07 2018

@author: hadoop
"""

from MySQLConnector import MySQLConnector
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def evalueHFBT(realGasSeries,realHFBT,predictHFBT,index):
    # 评估一个控制周期内燃气量的波动，此变量的量纲是100
    realGasSeriesStd = np.std(realGasSeries)
    RGSS_Scalar = 50.0
    realGasSeriesStd = realGasSeriesStd/RGSS_Scalar
    if index == 1:
        print(u'上一个控制周期(子模型：%d)，燃气序列的标准差为：%.4f'%(index,realGasSeriesStd))
    # 评估一个控制周期内预测回水温度和实际回水温度之间的相关性，此变量量纲为1
    RTP_Corr = np.corrcoef(predictHFBT,realHFBT)[0,1]
    first = np.mean(realHFBT[:int(0.3*len(realHFBT))])
    second = np.mean(realHFBT[int(0.3*len(realHFBT)):int(0.6*len(realHFBT))])
    third = np.mean(realHFBT[int(0.6*len(realHFBT)):int(1*len(realHFBT))])
    if abs(second - first) < 0.09 or abs(third - second) < 0.09:
        Real_SCorr = 0
    else:
        Real_SCorr = 1
    RTPC_Scalar = 1.0
    RTP_Corr = RTP_Corr/RTPC_Scalar
    # 评估一个控制周期内预测回水温度和实际回水温度之间的均方误，此变量量纲为0.1 
    RTP_MSE = np.mean((predictHFBT-realHFBT)**2)
    RTPM_Scalar = 0.1
    RTP_MSE = RTP_MSE/RTPM_Scalar
    # 一般来说，燃气标准差越大，越能容忍更高的误差
    if Real_SCorr:
        score = min(max(1,realGasSeriesStd),1.5)*(0.2*RTP_Corr + 0.8*(1-min(RTP_MSE,1)))
    else:
        print(u'触发设定条件，评估准则变更')
        score = min(max(1,realGasSeriesStd),1.5)*(1*(1-min(RTP_MSE,1)))
    print(u'上一个控制周期(子模型：%d)，预测曲线准确率评分为：%.4f'%(index,score))
    return min(1,score)

def OutterChecker(predictGroups,newData,downGasAmount,upGasAmount,logDir,controlLoops):
    
    # 基于起始和终止日期，建立合适的时间索引和DataFrame
    validDataFrame = pd.DataFrame(index = newData.index)
    realGasDataFrame = newData[['瞬时流量']]
    realGasDataFrame = realGasDataFrame.resample('1min').mean()
    validDataFrame[u'瞬时流量'] = realGasDataFrame[u'瞬时流量']
    realGasMean = np.mean(realGasDataFrame.as_matrix().ravel())
    # 查询换热站一次回温度
    realHFBTSeries = newData['一次回温度']
    realHFBTSeries = realHFBTSeries.resample('1min').mean()
    validDataFrame[u'一次回温度'] = realHFBTSeries
    # 确定预测组的恰当索引
    try:
        assert realGasMean > downGasAmount
        assert realGasMean < upGasAmount
        predictGroupsIndex  = realGasMean//100 - downGasAmount//100
    except AssertionError:
        if realGasMean <= downGasAmount:
            predictGroupsIndex = 0
        else:
            predictGroupsIndex = -1
    predictGroupsIndex = int(predictGroupsIndex)
    # 读取预测数据集
    def readPredictData(fileName):
        with open(fileName,'r') as f:
            ftext = f.read()
            data = []
            for line in ftext.split('\n'):
                if line == '':
                    continue
                data.append([float(i) for i in line.split(',')])
            return np.array(data)
    p_1 = readPredictData('P_1.txt')
    p_2 = readPredictData('P_2.txt')
    p_3 = readPredictData('P_3.txt')
    p_4 = readPredictData('P_4.txt')

    index = 1
    targetPredict = p_1[predictGroupsIndex]
    targetPredict_1 = targetPredict
    validDataFrame[u'预测回水温度'] = targetPredict
    validDataFrame = validDataFrame.fillna(method='ffill').fillna(method='bfill')
    score_1 = evalueHFBT(validDataFrame[u'瞬时流量'].as_matrix().ravel(),validDataFrame[u'一次回温度'].as_matrix().ravel(),
                        validDataFrame[u'预测回水温度'].as_matrix().ravel(),index)
    with open(u'model_1.txt','a') as f:
        f.write('%.2f\n'%score_1)

    index = 2
    targetPredict = p_2[predictGroupsIndex]
    targetPredict_2 = targetPredict
    validDataFrame[u'预测回水温度'] = targetPredict
    validDataFrame = validDataFrame.fillna(method='ffill').fillna(method='bfill')
    score_2 = evalueHFBT(validDataFrame[u'瞬时流量'].as_matrix().ravel(),validDataFrame[u'一次回温度'].as_matrix().ravel(),
                        validDataFrame[u'预测回水温度'].as_matrix().ravel(),index)
    with open(u'model_2.txt','a') as f:
        f.write('%.2f\n'%score_2)

    index = 3
    targetPredict = p_3[predictGroupsIndex]
    targetPredict_3 = targetPredict
    validDataFrame[u'预测回水温度'] = targetPredict
    validDataFrame = validDataFrame.fillna(method='ffill').fillna(method='bfill')
    score_3 = evalueHFBT(validDataFrame[u'瞬时流量'].as_matrix().ravel(),validDataFrame[u'一次回温度'].as_matrix().ravel(),
                        validDataFrame[u'预测回水温度'].as_matrix().ravel(),index)
    with open(u'model_3.txt','a') as f:
        f.write('%.2f\n'%score_3)
    
    index = 4
    targetPredict = p_4[predictGroupsIndex]
    targetPredict_4 = targetPredict
    validDataFrame[u'预测回水温度'] = targetPredict
    validDataFrame = validDataFrame.fillna(method='ffill').fillna(method='bfill')
    score_4 = evalueHFBT(validDataFrame[u'瞬时流量'].as_matrix().ravel(),validDataFrame[u'一次回温度'].as_matrix().ravel(),
                        validDataFrame[u'预测回水温度'].as_matrix().ravel(),index)
    with open(u'model_4.txt','a') as f:
        f.write('%.2f\n'%score_4)

    index = 5
    targetPredict = predictGroups[predictGroupsIndex]
    validDataFrame[u'预测回水温度'] = targetPredict
    validDataFrame = validDataFrame.fillna(method='ffill').fillna(method='bfill')
    score_5 = evalueHFBT(validDataFrame[u'瞬时流量'].as_matrix().ravel(),validDataFrame[u'一次回温度'].as_matrix().ravel(),
                        validDataFrame[u'预测回水温度'].as_matrix().ravel(),index)
    GasDis = [1]
    score = np.array([score_1,score_2,score_3,score_4])
    score = [1 if i >= 0.9 else i for i in score]
    score = [0.1 if i <= 0 else i for i in score]
    score = np.array(score)
    print(u'在该控制周期内实际耗费燃气量为：%.1f'%realGasMean)
    with open(os.path.join(logDir,u'scores.txt'),'w') as f:
        s = ','.join(['%.3f'%i for i in score])
        f.write(s)
    plotPredictGroups(predictGroups,controlLoops,validDataFrame[u'一次回温度'].as_matrix().ravel(),
                      validDataFrame[u'预测回水温度'].as_matrix().ravel(),[targetPredict_1,targetPredict_2,targetPredict_3,targetPredict_4])
    return GasDis,score_5
            
def plotPredictGroups(predictGroups,controlLoops,rBT,pBT,pBTs):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    for i in predictGroups:
        ax.plot(i,lw=0.8,c='r',ls='--')
    ax.plot(rBT,lw=1.5,c='k',ls='-',label='Reality')
    ax.plot(pBT,lw=1.5,c='blue',ls='-',label='Prediction')
    for indi in pBTs:
        ax.plot(indi,lw=1.1,c='green',ls='-.',zorder=200,alpha=0.7)
    ax.legend(loc='best')
        
    if not os.path.exists('PredictGroups'):
        os.mkdir('PredictGroups')
    figure.savefig('PredictGroups/第%d个控制周期预测.png'%controlLoops,dpi=300)
    np.savetxt('PredictGroups/第%d个控制周期预测序列.txt'%controlLoops,pBT,fmt='%.2f')
    np.savetxt('PredictGroups/第%d个控制周期实际序列.txt'%controlLoops,rBT,fmt = '%.2f')
    plt.close('all')
    
