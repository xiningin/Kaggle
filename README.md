# Kaggle Project
这里是一些在Kaggle上面学习过的项目，目前有：

### 1.Natural Language Processing with Disaster Tweets
&ensp;&ensp;&ensp;&ensp;二分类预测tweets是否有关真实灾难,为Kaggle自然语言处理的一个Getting Started Prediction Competition。[Natrural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

&ensp;&ensp;&ensp;&ensp;在问题的解决中对tweets数据做数据清洗，特征提取后利用prompt拼接所有特征信息，后续使用huggingface的Bert
预训练模型，并在最后添加一层全连接层作为分类头，达到分类预测的目的。
### 2.Google AI4Code – Understand Code in Python Notebooks
&ensp;&ensp;&ensp;&ensp;对于jupyter notebook的json文件中code cell顺序为markdown cell排序。[Google AI4Code – Understand Code in Python Notebooks](https://www.kaggle.com/competitions/AI4Code)

&ensp;&ensp;&ensp;&ensp;参考的baseline模型，仅通过模型对markdown cell的理解以及其顺序rank
的联系使用huggingface的DistilBert预训练模型接全连接层输出对每条markdown cell的百分比顺序(pct_rank)的预测。

&ensp;&ensp;&ensp;&ensp;但是此方法只使用了markdown cell中的文字信息，并未结合code cell中的数据信息后续考虑在利用排序好的code cell信息上改进，使用codeBert模型提取所有code cell提供的信息摘要添加入每条tokenize后的markdown cell中共同参与训练。目前决定使用huggingface提供的“codet5-base-multi-sum”模型生成代码摘要，但是代码需要从所有code cell中聚合并且清洗去除“\n”与部分注释等。

&ensp;&ensp;&ensp;&ensp;后续更新使用了codeBert模型代替原本的DistilBert模型，并且对于每个notebook截取一定数量(默认20)code在tokenizer处拼接加入，使模型具有对代码的认识，并以此来对markdown cell生成pct_rank。除此之外原本计划的使用codeT5模型生成代码摘要易遇到GPU或CPU内存不足、存在模型输入超过512个token被截断、生成总结效果不好等问题，且并未对codeBert的方法做根本性的优化，后续并未继续推进此方法。

&ensp;&ensp;&ensp;&ensp;代码和数据见AI4Code文件夹，其中train文件夹中的json文件较多，在此并未上传。
