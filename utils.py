import time
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import requests
import json
from sklearn.preprocessing import normalize

CITY_CODE_TO_CITY_INDEX_MAP = {
    '410100': 0,    # 郑州
    '410300': 1,    # 洛阳
    '410700': 2,    # 新乡
    '411000': 3,    # 许昌
}

DATA_TYPE = 'float64'


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

def _normalize(array, norm=None, return_norm=True, axis=1):
    """
    正则化
    x' = (x - mu) / sigma
    :param array:
    :param axis: 0 or 1
    :return: array_normed, mu, sigma
    """
    if array.ndim not in (2, 3):
        raise TypeError('当前正则化只支持2维或3维')
    if array.ndim == 3:
        x, y, z = array.shape
        _array = array.reshape((x * y, z))
        mu = np.mean(_array, axis=0)
        sigma = np.sqrt(np.var(_array, axis=0))
        _array_normed = (_array - mu) / sigma
        array_normed = _array_normed.reshape((x, y, z))
    else:
        mu = np.mean(array, axis=0)
        sigma = np.sqrt(np.var(array, axis=0))
        array_normed = (array - mu) / sigma
    return array_normed, mu, sigma


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
                raise TypeError('%s 不在410100, 410300, 410700, 411000中')
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
    return [_[:i] for _ in _[:-1]], _[-1][:i], [_[i:] for _ in _[:-1]], _[-1][i:]


def _shuffle(arrays):
    """
    :param arrays:
    :return:
    """
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


def draw_model(model):
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
    rsp = requests.get(url)
    cities_data = json.loads(rsp.text)
    with open('data.txt', mode='w', encoding='utf-8') as f:
        for city_data in cities_data:
            f.write(' '.join([str(city_data[key]) for key in city_data.keys()]) + '\n')


def get_r