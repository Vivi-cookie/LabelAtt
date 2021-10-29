# LabelAtt
基于标签嵌入注意力机制的多任务文本分类模型
## 环境要求
- python = 3.7
- pytorch = 1.7
- CUDA = 9.1
- 在.../wordvectors/pretrain_pycache_路径下存储golve模型预训练词向量，可以去https://nlp.stanford.edu/projects/glove/ 下载，有有50维、100维、300维三种。

## 代码框架
### Config.py: 参数设置
隐藏层数；batch_size；LSTM层数；训练次数；isRand：是否使用随机生成的词嵌入；isStatic：使用Glove词嵌入模型；isElmo：使用Elmo词嵌入模型 
### Tools.py: 基础函数的调用
- loadWordVectors（）：获得词嵌入
- saveCSVResult（）：保存模型运行结果
- loadDatsetByType(datasetType)：获得训练集、测试集
- create_variable（）：生成张量
- delMiddleContentWeightFiles(datasetType,classifierType):清除之前生成的中间文件内容（文本表示、标签分类、注意力权重）
- ....
### DatasetFactory.py: 存储、载入各数据集的dataloader类
### ModelFactory.py: 存储、载入各分类器模型类
### attention_visualize.py
- 输入：文本内容和文本注意力权重文件
- 输出：生成文本内容权重的可视化分析图。 
### main.py:主函数文件
- run_model_with_hyperparams(超参数、数据类别、分类器类别):跑模型
- train（）:训练集上迭代训练，每一batch训练后有误差反传和迭代器优化，获得train  accuracy和train loss
- test（）：在测试集上迭代训练，获得准确率、召回率、精确率和F1值
### wordvector文件夹:
- pretrain_wordvectors.py：生成word2vec词嵌入
- static_vectors.py：生成Glove词嵌入
- elmo_wordvectors.py:生成Elmo词嵌入
### dataset文件夹：
- TREC数据集
- SST1数据集
- CR数据集
### dataloader文件夹：各数据集生成相应的dataloader类
- TREC_Loader.py：生成TREC数据集训练集、测试集类
- SST1_Loader.py：生成SST1数据集训练集、测试集类
- CustomerReview_Loader.py: 生成CR数据集训练集、测试集类
- preprocess.py ：数据预处理，即对每一行输入数据进行清洗
### models文件夹：各种分类器模型
-	LSTM_IAtt.py: LSTM_IAtt模型
-	LSTM_Attention.py：LSTMAtt模型
-	SelfAttention.py: SelfAtt模型
-	FTIA.py: FTIA模型
### TextRepresentations文件夹
-	 文本标签：'tSNE_' + dataType + '_' + classifierType + '_labels.txt'文件
-	 文本表示：'tSNE_' + dataType + '_' + classifierType + '_textRepr.txt'文件
-	 文本内容：datasetType + '_' + classifierType + '_contents.txt文件
-	 文本注意力权重：datasetType + '_' + classifierType + '_attweights.txt'文件

### visualization文件夹：
-	tSNE_implementation.py：使用tsne降维，导入生成的文本表示和分类标签生成相应的散点图、进行可视化分析。
### figs文件夹：存放生成的可视化图片

### 实验结果
- 注意力权重分析：
![LSTMAtt和LabelAtt在TREC数据集上的注意力权重分布对比示例](https://user-images.githubusercontent.com/65707124/139387601-02f2c896-0d41-4720-8faa-20a6291c65d2.png) 

LSTMAtt和LabelAtt在TREC数据集上的注意力权重分布对比示例

![image](https://user-images.githubusercontent.com/65707124/139388297-84cd3310-ef26-4440-b0fd-654ffd45c134.png)

LSTMAtt和LabelAtt在CR数据集上的注意力权重分布对比示例 
- 文本聚类可视化分析：
-![image](https://user-images.githubusercontent.com/65707124/139387543-72f046a1-df27-455b-bc76-ad43efd5a08d.png)
![image](https://user-images.githubusercontent.com/65707124/139388371-b210466c-c541-4d4c-9307-736c49c0813b.png)

