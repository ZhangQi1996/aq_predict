from keras.layers import LSTM, SimpleRNN, Input, Add
from utils import *


def get_model(is_save_model_struct_img=False):
    """
    获取AQPredict模型
    """
    inputs_zz = Input(shape=(6,), batch_shape=(32, 24, 6))
    inputs_xx = Input(shape=(6,), batch_shape=(32, 3, 6))
    inputs_ly = Input(shape=(6,), batch_shape=(32, 3, 6))
    inputs_xc = Input(shape=(6,), batch_shape=(32, 3, 6))

    # 关于return_sequences，return_state的设置参见https://blog.csdn.net/u011327333/article/details/78501054
    # lstm_layer_zz_outputs = (lstm_output, state_h, state_c) 这里lstm_output==state_h
    # stateful用来控制当前state_h, state_c的值是否为上一批次所得的states值
    lstm_layer_zz_outputs = LSTM(units=6, input_shape=(24, 6), return_sequences=False, return_state=True, stateful=False)(inputs_zz)
    rnn_layer_xx_outputs = SimpleRNN(units=6, input_shape=(3, 6), return_sequences=False, return_state=True, stateful=False)(inputs_xx)
    rnn_layer_ly_outputs = SimpleRNN(units=6, input_shape=(3, 6), return_sequences=False, return_state=True, stateful=False)(inputs_ly)
    rnn_layer_xc_outputs = SimpleRNN(units=6, input_shape=(3, 6), return_sequences=False, return_state=True, stateful=False)(inputs_xc)

    encoder_outputs = Add()([lstm_layer_zz_outputs[0], rnn_layer_xx_outputs[0], rnn_layer_ly_outputs[0], rnn_layer_xc_outputs[0]])

    # 拼装states, 合并hstm_layer_zz, rnn_layer_xx, rnn_layer_ly, rnn_layer_xc层的state_h值(即output值，由于二者相等)
    # 而state_c的值为hstm_layer_zz的state_c的值
    encoder_states = [encoder_outputs, lstm_layer_zz_outputs[-1]]

    decoder = LSTM(6, input_shape=(4, 6), return_sequences=True, activation='linear')

    # 全零
    decoder_inputs = Input(shape=(6,), batch_shape=(32, 4, 6))

    decoder_outputs = decoder(decoder_inputs, initial_state=encoder_states)
    model = Model(inputs=[inputs_zz, inputs_xx, inputs_ly, inputs_xc, decoder_inputs], outputs=decoder_outputs)
    model.compile(loss='mse', optimizer='adam')
    if is_save_model_struct_img:
        draw_model(model)
        print("保存模型结构图成功...")
    # conf_json = model.to_json()
    # save_model_structure(model, format='yaml')
    return model