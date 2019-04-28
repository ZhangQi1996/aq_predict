import time
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import requests
import json

CITY_CODE_TO_CITY_INDEX_MAP = {
    '410100': 0,    # 郑州
    '410300': 1,    # 洛阳
    '410700': 2,    # 新乡
    '411000': 3,    # 许昌
}


def _repair_data(data, time_gap = 3600):
    """
    data只能为zz_data, xx_data, ly_data, xc_data其中任意一个
    修复data中的非法数据项
    e.g. 数据项 [1553162400, 66.0, 12.0, 81.0, 4.0, 26.0, 0.4, 67.0]
    :return [66.0, 12.0, 81.0, 4.0, 26.0, 0.4, 67.0]
    """
    i = 0
    data_len = len(data)
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
            data.insert(i, data[i-1])
            data_len += 1
        i += 1
    return data


def _data_loader(file_name='data.txt', encoding='utf-8', ):
    """
    返回四个城市的数据列表
    每个数据项为：时间(秒值)_AQI_PM2.5_PM10_SO2_NO2_CO_O3
    e.g. [1553162400, 66.0, 12.0, 81.0, 4.0, 26.0, 0.4, 67.0]
    :param file_name:
    :param encoding:
    :return:
    """
    zz_data = []    # 郑州
    xx_data = []    # 洛阳
    ly_data = []    # 新乡
    xc_data = []    # 许昌
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
                zz_data.append(data[1: -1])
            elif data[0] == '410300':
                ly_data.append(data[1: -1])
            elif data[0] == '410700':
                xx_data.append(data[1: -1])
            elif data[0] == '411000':
                xc_data.append(data[1: -1])
            else:
                raise TypeError('%s 不在410100, 410300, 410700, 411000中')
    return zz_data, xx_data, ly_data, xc_data


def data_loader(file_name='data.txt', encoding='utf-8', ratio=0.7):
    """
    外部调用此函数
    :param file_name:
    :param encoding:
    :param ratio: 数据集中训练跟预测数据比
    :return:
    """
    zz_data, xx_data, ly_data, xc_data = [_repair_data(data) for data in _data_loader(file_name, encoding)]
    begin_t = zz_data[0][0]  # 获取开始时间
    data_len = len(zz_data)
    if data_len < 24:
        raise Exception('所提供的的数据项长度应该大于等于24，你所提供的数据项长度为%s' % data_len)
    # 郑州每组取24个数据，其他每个取3个数据
    i = 0
    step_span = 1
    while i < data_len:
        for j in range(i, i + 24):
            data_item = zz_data[j][1:]  # 郑州


        i += step_span
        if i + 24 > data_len:  # 下一轮越界
            break


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

