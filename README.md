# Mutimodal-Relation-Extraction(Course Design)
## 前言
&emsp;&emsp;本仓库是为了记录之前课程设计的收获，也是为了以后有机会更够更进一步完善它。由于是首次使用github真正创建一个仓库，可能还有不完善的地方。本项目代码是基于论文[Multimodal Relation Extraction with Efficient Graph Alignment](https://dl.acm.org/doi/abs/10.1145/3474085.3476968)及其[项目代码](https://github.com/thecharm/Mega)实现的,并作出了一定修改。
## 编程环境
1. 编程平台：vscode远程连接linux服务器方式
2. 编程语言：python >= 3.8，javascript(用于实现网页,在具体运行模型时可以不用管)
3. python依赖的库详见文件[requirements.txt](requirements.txt)
4. 机器要求：训练时需要使用GPU，GPU Memory >= 20G
## 文件说明
1. benchmark文件夹
    - 用于存放[mega数据集](https://drive.google.com/file/d/1FYiJFtRayWY32nRH0rdycYzIdDcMmDFR/view)，由于数据集过大，此处不存放
    - 支持网页所需要的js文件、网页代码和网页端调用模型的python文件
2. ckpt文件夹
    - 用于保持和原项目文件夹一致
3. example文件夹
    - 存放模型训练文件，train_supervised_bert.py
    - web_page.py，精简化的代码，只有网页调用时所需要的模型，在此处对dataloader和模型进行了一定的修改
4. Opennre文件夹
    - 存放模型使用的编码器和特征提取实现的代码
    - 存放具体使用的模型，决定得到最终结果的方式
