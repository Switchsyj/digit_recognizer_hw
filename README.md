# digit_recognizer_hw
A homework project for hand-written digit recognizer(0-9) using DNN
***
version: python3.6 + tensorflow 1.2
***
```text
.
|---- MINIST_data (MINIST数据集)
|---- model (tensorflow保存的checkpoint)
|---- input_data.py (通过keras加载MINIST数据集 转换onehot label)
|---- deep_neural_network.py (手动实现DNN forward和backward)
|---- train_model.py (手写DNN的实现(准确率比较感人))
|---- learning_rate_decay (实现学习率下降)
|---- dnn_utils.py (激活函数及其对应的求导函数)
——————————————————————————————
|---- input_data_tensorflow.py (通过下载官方数据 手动实现minibatch)
|---- tensorflow_dnn.py (tensorflow实现DNN 调用opencv实现实时识别)
```