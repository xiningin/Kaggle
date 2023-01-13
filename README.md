# Kaggle Project
这里是一些在Kaggle上面学习过的项目，目前有：

### 1.Natural Language Processing with Disaster Tweets
&ensp;&ensp;&ensp;&ensp;二分类预测tweets是否有关真实灾难,为Kaggle自然语言处理的一个Getting Started Prediction Competition。

&ensp;&ensp;&ensp;&ensp;在问题的解决中对tweets数据做数据清洗，特征提取后利用prompt拼接所有特征信息，后续使用huggingface的Bert
预训练模型，并在最后添加一层全连接层作为分类头，达到分类预测的目的。
### 2.Google AI4Code – Understand Code in Python Notebooks
&ensp;&ensp;&ensp;&ensp;对于jupyter notebook的json文件中code cell顺序为markdown cell排序。

&ensp;&ensp;&ensp;&ensp;参考的baseline模型，仅通过模型对markdown cell的理解以及其顺序rank
的联系使用huggingface的DistilBert预训练模型接全连接层输出对每条markdown cell的百分比顺序(pct_rank)的预测。

&ensp;&ensp;&ensp;&ensp;但是此方法只使用了markdown cell中的文字信息，并未结合code cell中的数据信息后续考虑在利用排序好的code cell信息上改进，使用codeBert模型提取所有code cell提供的信息摘要添加入每条tokenize后的markdown cell中共同参与训练。
