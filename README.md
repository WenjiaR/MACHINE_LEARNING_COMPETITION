# 需说明的情况

- 注意数据文件中test, eval和train都包含两层相同的文件名，与作业要求的源代码有所不同，例如需要检索到训练集中的一张图片
，路径为./data/food/train/train/0/1711.jpg, train出现了两次。

- 具体的data文件已被.gitignore。
- 图片路径索引文件请参见./data/food/test.txt, ./data/food/train.txt,./data/food/val.txt

# 代码效果
- Kaggle一次提交得分为0.46509

# 测试文件编写基本思路
首先先生成测试集图片路径的索引保存在test.txt中，然后用实现数据集DataSet，利用数据集实现dataloader，
使用dataloader可实现测试数据集的遍历，对每一张图片进行预测，输出结果。
