from utils import data_loader, draw_train_loss_curve, standardize, save_std_params_list_or_params,_standardize
from model import get_model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = data_loader()
    # 对train_x归一化
    train_x, params_list = standardize(train_x)
    test_x = standardize(test_x, params_list=params_list, return_params_list=False)
    train_y, params = _standardize(train_y)
    test_y = _standardize(test_y, params=params, return_params=False)
    model = get_model()
    history = model.fit(x=train_x, y=train_y, batch_size=32, verbose=2, epochs=3000, validation_data=(test_x, test_y))
    model.save_weights(filepath='model_weights.h5')
    save_std_params_list_or_params(params_list_or_params=params_list, file_name='x_params_list.txt', is_params_list=True)
    save_std_params_list_or_params(params_list_or_params=params, file_name='y_params.txt', is_params_list=False)
    draw_train_loss_curve(history=history)
