from math import sqrt
import numpy
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


# 将时间序列转换为监督式学习问题
#data：作为列表或2D NumPy数组的观察序列。需要。
#n_in：作为输入（X）的滞后观测值的数量。值可能介于[1..len（data）]可选。默认为1。
#n_out：作为输出的观察次数（y）。值可以在[0..len（data）-1]之间。可选的。默认为1。
#dropnan：布尔型是否删除具有NaN值的行。可选的。默认为True。
#该函数返回单个值：
#返回：为监督学习构筑的Pandas DataFrame系列
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print('带转换数据')
    print(df.head())
    cols, names = list(), list()

    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        print('shift数据')
        print(cols[0][0:5])
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    print('names数据')
    print(names[0:5])
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:#t时刻
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #拼接
    agg = concat(cols, axis=1)
    agg.columns = names
    print("拼接")
    print(agg[0:5])
    # #将空值NaN行删除
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 加载数据
dataset = read_csv('EURUSD_1_Min.csv', header=0, index_col=0)
#dataset = read_csv('Rate.csv', header=0, index_col=0)
values = dataset.values
print('原始数据')
print(values[0:5] )
##由于4列的风向是标签，编码成整数
#encoder = LabelEncoder()#简单来说 LabelEncoder 是对不连续的数字或者文本进行编号
#values[:, 4] = encoder.fit_transform(values[:, 4])
#使所有数据是float类型
values = values.astype('float32')
# 所有特征归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 将数据集转换为监督学习问题
reframed = series_to_supervised(scaled, 1, 1)
#然后去除待预测小时的天气变量（t）。#删除不预测的列
reframed.drop(reframed.columns[[1, 2, 3, 4, 9]], axis=1, inplace=True)
print(reframed.head())

# 分成训练集和测试集(用第1年作为训练集，剩余4年为测试集)
values = reframed.values
#n_train_hours = 24 * 60 *2
n_train_hours = int(len(dataset) * 0.63)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 将训练集和测试集分解为输入和输出变量(train_y为当前污染值，train_x为前一小时污染数据)
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 输入（X）被重新整形为LSTM预期的3D格式，即[采样，时间步长，特征]。
#样品：一个序列是一个样本； 时间步：一个时间步代表样本中的一个观察点；特征：一个特征是在一个时间步长的观察得到的s
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))#每次输入一个8维的一个向量
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 设计网络
#用第一隐层中的50个神经元和输出层中的1个神经元来定义LSTM，以预测污染。
model = Sequential()
# model.add(LSTM(6, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))#输入形状将是8个特征的1个时间步(一组八维的向量)
# model.add(LSTM(150, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))#在最后一个LSTM层中，参数返回序列应该是假的，以避免与密集层的输入维度不兼容
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # 训练网络
# history = model.fit(train_X, train_y, epochs=100, batch_size=60, validation_data=(test_X, test_y), verbose=2,
#                     shuffle=False)
# # 绘制训练和测试损失
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#模型保存
#model.save('ExchangaRateModel_80.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
#model = load_model('ExchangaRateModel_63.h5')

#模型拟合后，我们可以预测整个测试数据集
# 做预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#通过预测和实际值的原始比例，我们可以计算模型的误差分数。
# 在这种情况下，我们计算出与变量本身相同单位给出误差的均方根误差（RMSE）。
#预测数据逆缩放
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
#真实数据逆缩放
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# 计算RMSE均方根误差：预测值与真值偏差的平方和观测次数n比值的平方根
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.6f' % rmse)


pyplot.plot(yhat)
pyplot.plot(test_y)
pyplot.legend()
pyplot.show()
