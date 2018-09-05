# Text classification with CNN and Word2vec
本文是参考gaussic大牛的“text-classification-cnn-rnn”后，基于同样的数据集，嵌入词级别所做的CNN文本分类实验结果，gaussic大牛是基于字符级的；<br><br>
本实验的主要目是为了探究基于Word2vec训练的词向量嵌入CNN后，对模型的影响，实验结果得到的模型在验证集达到96.5%的效果，gaussic大牛为94.12%；<br><br>
更多详细可以阅读gaussic大牛的博客：[text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)<br><br>

1 环境
=
python3<br>
tensorflow 1.3以上<br>
gensim<br>
jieba<br>
scipy<br>
numpy<br>
scikit-learn<br>

2 CNN卷积神经网络
=

3 数据集
=

4 预处理
=

5 运行步骤
=

6 训练结果
=

7 测试结果
=

8 预测结果
=
运行:python text_predict.py <br>
随机从测试数据中挑选了五个样本，输出原文本和它的原文本标签和预测的标签，下图中5个样本预测的都是对的；<br>
![image](https://github.com/cjymz886/text-cnn/blob/master/images/text_cnn_predict.png)


9 参考
=
1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2. [gaussic/text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
3. [YCG09/tf-text-classification](https://github.com/YCG09/tf-text-classification)

