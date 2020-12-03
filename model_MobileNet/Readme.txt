1. TK.py是GUI软件的源代码
2. mobile.py是进行网络训练的代码
3. test_random.py是循环随机抽取测试集图片并打印整体准确率和错误种类的代码
4. test_single.py是单张图片进行预测的代码
5. class_indices.json中是数据标签索引（mobile.py运行后自行生成）
6. mobilenet.h5是Mobile net网络模型的框架结构以及参数（mobile.py运行后自行生成）
7. test中存放的是花卉测试数据集的图像（自己创建）
8. train中存放的是花卉训练数据集的图像（自己创建）
（test和train中已包含所有已处理后的花卉数据集，并且考虑存储空间，故不再展示数据集处理过程及代码）