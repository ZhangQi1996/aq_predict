import time
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import requests
import json
from sklearn.metrics import r2_score    #R square
from keras import backend as K

CITY_CODE_TO_CITY_INDEX_MAP = {
    '410100': 0,    # 郑州
    '410300': 1,    # 洛阳
    '410700': 2,    # 新乡
    '411000': 3,    # 许昌
}

DATA_TYPE = 'float32'


def _col_combine_vec(target_array, vec):
    """
    将vec向量列合并到目标数组中
    :param target_array:
    :param vec: 可以是list, np.ndarray, 可以是横向量，可以是纵向量
    :return:
    e.g.
    target_array = [
        [1, 2],
        [4, 5],
    ]
    vec = [3, 6]
    rst ==> [
        [1, 2, 3],
        [4, 5, 6],
    ]
    """
    if isinstance(vec, list):
        vec = np.array(vec, dtype=DATA_TYPE)
    if isinstance(target_array, list):
        target_array = np.array(target_array, dtype=DATA_TYPE)
    if target_array.size == 0:
        return vec.reshape(*[1 for _ in range(target_array.ndim - 2)], len(vec), 1)
    return np.concatenate((target_array, vec.reshape(*[1 for _ in range(target_array.ndim - 2)], len(vec), 1)),
                          axis=target_array.ndim - 1)


def _row_combine_vec(target_array, vec):
    """参见_col_combine_vec"""
    if isinstance(vec, list):
        vec = np.array(vec, dtype=DATA_TYPE)
    if isinstance(target_array, list):
        target_array = np.array(target_array, dtype=DATA_TYPE)
    if target_array.size == 0:
        return vec.reshape(*[1 for _ in range(target_array.ndim - 1)], len(vec))
    return np.concatenate((target_array, vec.reshape(*[1 for _ in range(target_array.ndim - 1)], len(vec))),
                          axis=target_array.ndim - 2)


def _repair_data(data, time_gap=3600):
    """
    data只能为zz_data, xx_data, ly_data, xc_data其中任意一个
    修复data中的非法数据项
    e.g. 数据项 [1553162400, 66.0, 12.0, 81.0, 4.0, 26.0, 0.4, 67.0]
    :return [66.0, 12.0, 81.0, 4.0, 26.0, 0.4, 67.0]
    """
    i, data_len = 0, len(data)
    while i < data_len:
        # 修复数据值非法
        if data[i][1] < 5:
            if i == 0:
                if data[1][1] < 5:
                    raise ValueError('所提供的的data不能前两项都不合法。。')
                else:
                    data[0] = data[1]
            else:
                data[data_len-1] = data[data_len-2]
        # 修复数据项缺失
        if i != 0 and data[i][0] - data[i-1][0] > time_gap:
            data = np.concatenate((data[:i], data[i-1].reshape(1, *data[i-1].shape), data[i:]), axis=0)
            data[i][0] += time_gap
            data_len += 1
        i += 1
    return np.concatenate([data[i][2:].reshape((1, 6)) for i in range(data_len)], axis=0)


def _standardize(array, params=None, return_params=True):
    """
    调整每个属性到X~N(0, sigma)
    x' = (x - mu) / sigma
    :param array:
    :param params: (mu, sigma)
    :return: array_stded, mu, sigma
    """
    if array.ndim not in (2, 3):
        raise TypeError('当前正则化只支持2维或3维')
    if array.ndim == 3:
        x, y, z = array.shape
        _array = array.reshape((x * y, z))
        if params is None:
            mu = np.mean(_array, axis=0)
            sigma = np.sqrt(np.var(_array, axis=0))
        else:
            mu, sigma = params
        if mu.all() != 0 or sigma.all() != 0:
            _array_stded = (_array - mu) / sigma
            array_stded = _array_stded.reshape((x, y, z))
        else:
            array_stded = np.zeros((x, y, z))
    else:
        if params is None:
            mu = np.mean(array, axis=0)
            sigma = np.sqrt(np.var(array, axis=0))
        else:
            mu, sigma = params
        if mu.all() != 0 or sigma.all() != 0:
            array_stded = (array - mu) / sigma
        else:
            array_stded = np.zeros(array.shape)
    if return_params:
        return array_stded, (mu, sigma)
    else:
        return array_stded


def standardize(x, params_list=None, return_params_list=True):
    """将x标准化，其中x = [zz_inputs, xx_inputs, ly_inputs, xc_inputs, decoder_inputs]"""
    x_stded = []
    if params_list is not None:
        for i in range(len(x)):
            x_stded.append(_standardize(x[i], params=params_list[i], return_params=True)[0])
    else:
        params_list = []
        for i in range(len(x)):
            _ = _standardize(x[i], params=None, return_params=True)
            x_stded.append(_[0])
            params_list.append(_[1])
    if return_params_list:
        return x_stded, params_list
    else:
        return x_stded


def inverse_std(array_stded, params):
    """将已经标准化过的数组逆标准化，支持1~3维数据，常见x, y"""
    mu, sigma = params
    if array_stded.ndim < 3:
        return array_stded * sigma + mu
    elif array_stded.ndim == 3:
        x, y, z = array_stded.shape
        array = array_stded.reshape((x * y, z)) * sigma + mu
        return array.reshape((x, y, z))
    else:
        raise TypeError('不支持大于3维的数组逆标准化..')


def save_std_params_list_or_params(params_list_or_params, file_name, encoding='utf-8', is_params_list=True):
    with open(file=file_name, mode='w', encoding=encoding) as f:
        if is_params_list:
            params_list = params_list_or_params
            for params in params_list:
                mu, sigma = params
                f.write(' '.join([str(mu[i]) for i in range(len(mu))]) + '#')
                f.write(' '.join([str(sigma[i]) for i in range(len(sigma))]) + '\n')
        else:
            params = params_list_or_params
            mu, sigma = params
            f.write(' '.join([str(mu[i]) for i in range(len(mu))]) + '#')
            f.write(' '.join([str(sigma[i]) for i in range(len(sigma))]) + '\n')


def load_std_params_list_or_params(file_name, encoding='utf-8', is_params_list=True):
    params_list = []
    with open(file=file_name, mode='r', encoding=encoding) as f:
        while f.readable():
            line = f.readline()
            if line == '':
                break
            mu, sigma = line.rstrip('\n').split('#')
            mu = np.array([float(_) for _ in mu.split(' ')])
            sigma = np.array([float(_) for _ in sigma.split(' ')])
            params_list.append((mu, sigma))
    if is_params_list:
        return params_list
    else:
        return params_list[0]


def _data_loader(file_name='data.txt', encoding='utf-8', ):
    """
    返回四个城市的数据列表
    每个数据项为：时间(秒值)_AQI_PM2.5_PM10_SO2_NO2_CO_O3
    e.g. [1553162400, 66.0, 12.0, 81.0, 4.0, 26.0, 0.4, 67.0]
    :param file_name:
    :param encoding:
    :return:
    """
    zz_data = np.array([[]])    # 郑州
    xx_data = np.array([[]])    # 洛阳
    ly_data = np.array([[]])    # 新乡
    xc_data = np.array([[]])    # 许昌
    with open(file_name, mode='r', encoding=encoding) as f:
        while f.readable():
            line = f.readline()
            if line == '':
                break
            data = line.rstrip('\n').split(' ')
            data[1] = int(time.mktime(time.strptime(data[1], "%Y-%m-%dT%H:%M:%S")))
            data_len = len(data)
            for i in range(2, data_len - 1):
                data[i] = float(data[i])
            if data[0] == '410100':
                zz_data = _row_combine_vec(zz_data, data[1: -1])
            elif data[0] == '410300':
                ly_data = _row_combine_vec(ly_data, data[1: -1])
            elif data[0] == '410700':
                xx_data = _row_combine_vec(xx_data, data[1: -1])
            elif data[0] == '411000':
                xc_data = _row_combine_vec(xc_data, data[1: -1])
            else:
                raise TypeError('%s 不在410100, 410300, 410700, 411000中' % data[0])
    return zz_data, xx_data, ly_data, xc_data


def data_loader(file_name='data.txt', encoding='utf-8', ratio=0.7, decoder_input_units=4, step_span=1):
    """
    外部调用此函数
    :param file_name:
    :param encoding:
    :param ratio: 数据集中训练跟预测数据比
    :return: train_x, train_y, test_x, test_y
    train_x = [zz_inputs, xx_inputs, ly_inputs, xc_inputs, decoder_inputs]
    train_y = zz_outputs
    """
    print("loading data...")
    # 返回修复后的数据(要是数据本来就正常保持不变)
    zz_data, xx_data, ly_data, xc_data = [_repair_data(data) for data in _data_loader(file_name, encoding)]
    data_len = len(zz_data)
    if data_len < 24 + decoder_input_units:
        raise Exception('所提供的的数据项长度应该大于等于%s，你所提供的数据项长度为%s' % (24 + decoder_input_units, data_len))
    # 郑州每组取24个数据，其他每个取3个数据, 再取4个未来数据
    i = 0
    zz_inputs, zz_outputs, xx_inputs, ly_inputs, xc_inputs = [np.array([[[]]], dtype=DATA_TYPE) for _ in range(5)]
    while True:
        zz_input, zz_output, xx_input, ly_input, xc_input = [np.array([[]], dtype=DATA_TYPE) for _ in range(5)]
        for j in range(i, i + 24):
            zz_input = _row_combine_vec(zz_input, zz_data[j][:])	    # 郑州
        for j in range(i + 21, i + 24):		# 取最后三项
            xx_input = _row_combine_vec(xx_input, xx_data[j])	    # 新乡
            ly_input = _row_combine_vec(ly_input, ly_data[j])	    # 洛阳
            xc_input = _row_combine_vec(xc_input, xc_data[j])	    # 许昌
        for j in range(i + 24, i + 24 + decoder_input_units):   # 郑州未来
            zz_output = _row_combine_vec(zz_output, zz_data[j][:])

        def f(inputs, input):
            if inputs.size == 0:
                return input.reshape((1, *input.shape))
            return np.append(inputs, input.reshape((1, *input.shape)), axis=0)
        zz_inputs = f(zz_inputs, zz_input)
        xx_inputs = f(xx_inputs, xx_input)
        ly_inputs = f(ly_inputs, ly_input)
        xc_inputs = f(xc_inputs, xc_input)
        zz_outputs = f(zz_outputs, zz_output)
        i += step_span
        if i + 24 + decoder_input_units > data_len:  # 24 + decoder_input_units个预测，下一轮越界
            break
    data_len = len(zz_inputs)
    decoder_inputs = np.zeros(shape=(data_len, decoder_input_units, 6))
    _ = _shuffle((zz_inputs, xx_inputs, ly_inputs, xc_inputs, decoder_inputs, zz_outputs))
    i = int(data_len * ratio)
    print("在修复模式下总共读取%s条数据，其中训练集数据为%s条，测试集数据为%s条，当前比率为%s" % (data_len, i, data_len - i, ratio))
    return [_[:i] for _ in _[:-1]], _[-1][:i], [_[i:] for _ in _[:-1]], _[-1][i:]


def _shuffle(arrays):
    """
    :param arrays:
    :return:
    """
    print("正在shuffle数据...")
    l = len(arrays[0])
    for array in arrays:
        if len(array) != l:
            raise ValueError('每个数组的大小必须一致。。。')
    shuffle_list = [i for i in range(l)]
    np.random.shuffle(shuffle_list)
    ret = []
    for array in arrays:
        _ = np.zeros(array.shape)
        for i in range(l):
            _[i] = array[shuffle_list[i]]
        ret.append(_)
    print("shuffle数据完成...")
    return ret


def save_model_structure(model, file_name_prefix='conf', encoding='utf-8', format='json'):
    """
    只保存模型结构，加载时使用
    :param model: 一个Model实例
    :param file_name_prefix: 保存的文件名前缀
    :param encoding: 编码
    :param format: 保存的文件格式
    :return:
    warning: 像一些复杂的结构将无法被序列化（e.g. Add Layer）如果出现类似结构推荐使用model.save()
    """
    assert isinstance(model, Model), "model instance must be a Model instance."
    if format == 'json':
        conf_info_str = model.to_json()
    elif format == 'yaml':
        conf_info_str = model.to_yaml()
    else:
        raise ValueError('param: format must be json or yaml, %s is not supported.' % format)
    with open(file_name_prefix + '.' + format, mode='w', encoding=encoding) as f:
        f.write(conf_info_str)
    import warnings
    warnings.warn("像一些复杂的结构将无法被序列化（e.g. Add Layer）如果出现类似结构推荐使用model.save()")


def _draw_model(model):
    """
    绘制模型结构
    :param model:
    :return:
    """
    assert isinstance(model, Model), "model instance must be a Model instance."
    plot_model(model, show_shapes=True, rankdir='TR')


def create_data_txt_from_json_interface():
    """
    从提供的相关预测城市数据json数据接口中整理数据保存在data.txt中
    格式：
    城市码_时间_AQI_PM2.5_PM10_SO2_NO2_CO_O3_首要污染物
    :return:
    """
    url = "http://www.david-zhang.cn:8080/v1/cur_data/part/rlv_pred_cities/"
    print("正在从接口%s下载数据..." % url)
    rsp = requests.get(url)
    print("下载完成，正在转换数据并存入data.txt...")
    cities_data = json.loads(rsp.text)
    with open('data.txt', mode='w', encoding='utf-8') as f:
        for city_data in cities_data:
            f.write(' '.join([str(city_data[key]) for key in city_data.keys()]) + '\n')
    print("写入完成...")


def get_adjusted_r2_score(y_true, y_pred, return_str=False):
    """
    回归评价指标：校正决定系数（Adjusted R-Square）
    参见：https://blog.csdn.net/u012735708/article/details/84337262
    Adjusted R-Square 抵消样本数量对 R-Square的影响，做到了真正的 0~1，越大越好。
    :param y_true:
    :param y_pred:
    :return: 0~1的浮点值（越大越好）
    """
    assert y_true.shape == y_pred.shape, 'y_true 与 y_pred的维度不同'
    if y_pred.ndim == 3:
        simples_n, pred_n, features_n = y_pred.shape
        y_true = y_true.reshape((y_true.shape[0] * y_true.shape[1], y_true.shape[2]))
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2]))
    else:
        raise TypeError('所提供的y_true, y_pred都必须维度数 == 3, 而实际维度数为%s' % y_pred.ndim)
    simples_n *= pred_n
    _ = r2_score(y_true, y_pred)
    _ = (_ - 1) * (simples_n - 1) / (simples_n - features_n - 1) + 1
    if return_str:
        return '回归评价指标：校正决定系数（Adjusted R-Square）= %s (越靠近1越有效)' % _
    else:
        return _


def draw_train_loss_curve(history):
    """
    绘制训练得到的loss曲线
    :param history:
    :return:
    """
    from matplotlib import pyplot as plt
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def adjusted_mae(y_true, y_pred, penalize_1=3, penalize_2=10):
    """
    默认y_true和y_pred都是(simples, units, features)
    作为mae的基于本问题作出适当修正，增大第三和第五列的惩罚
    """
    _ = y_pred - y_true
    penalize_1 *= _[:][:][2]
    penalize_2 *= _[:][:][4]
    return K.mean(K.abs(_), axis=-1) + K.mean(K.abs(penalize_1 + penalize_2), axis=-1)
