from model import get_model
from utils import data_loader, get_adjusted_r2_score, load_std_params_list_or_params, standardize, inverse_std

def _predict_many(model, x_stded, y_params):
    """
    多条记录预测
    :param x_stded: 标准化后的x
    :param y_params: 将预测所得的y_pred_stded进行逆标准化所需的params = (mu, sigma)
    :return: y_pred
    """
    y_pred_stded = model.predict(x_stded)
    return inverse_std(array_stded=y_pred_stded, params=y_params)


def _predict_one(model, x_stded, y_params):
    """参见_predict_many函数"""
    if x_stded[0].ndim != 2:
        raise TypeError('所传入单样本x格式不合法...是否误传入多样本，错误维度:%s' % x_stded[0].ndim)
    x_stded = [_.reshape((1, *_.shape)) for _ in x_stded]
    return _predict_many(model, x_stded, y_params)


def predict(model, x, x_params_list, y_params, is_many=True):
    """
    支持单样本和多样本预测
    当为多样本时，x = [zz_inputs, xx_inputs, ly_inputs, xc_inputs, decoder_inputs]，其中.._inputs = (simples, units, features)
    当为多样本时，x = [zz_input, xx_input, ly_input, xc_input, decoder_input]，其中.._input = (units, features)
    :param model: AQPredict模型
    :param x:
    :param x_params_list:
    :param y_params:
    :param is_many: 是否为多样本
    :return: y_pred = (simples, features) or (1, features)
    """
    if is_many:
        x_stded = standardize(x, x_params_list, return_params_list=False)
        return _predict_many(model, x_stded, y_params)
    else:
        x_stded = standardize(x, x_params_list, return_params_list=False)
        return _predict_one(model, x_stded, y_params)


if __name__ == '__main__':
    model = get_model()
    model.load_weights('model_weights.h5')
    _, _, test_x, test_y = data_loader()
    x = [_[0] for _ in test_x]
    y = test_y[0]
    x_params_list = load_std_params_list_or_params(file_name='x_params_list.txt', is_params_list=True)
    y_params = load_std_params_list_or_params(file_name='y_params.txt', is_params_list=False)
    y_pred = predict(model=model, x=x, x_params_list=x_params_list, y_params=y_params, is_many=False)
    print(y)
    print(y_pred)

