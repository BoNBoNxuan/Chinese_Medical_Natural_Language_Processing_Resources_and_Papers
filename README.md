# Chinese Medical Natural Language Processing Resources and Papers

* [Chinese Medical Natural Language Processing_Resources](#chinese_medical_natural_language_processing_resources)
  * [中文医疗数据集](#中文医疗数据集)
    * [1.Yidu-S4K：医渡云结构化4K数据集](#1yidu-s4k医渡云结构化4k数据集)
    * [2.Yidu-N7K：医渡云标准化7K数据集](#2yidu-n7k医渡云标准化7k数据集)
    * [3.瑞金医院MMC人工智能辅助构建知识图谱大赛](#3瑞金医院mmc人工智能辅助构建知识图谱大赛)
    * [4.中文医药方面的问答数据集](#4中文医药方面的问答数据集)
    * [5.平安医疗科技疾病问答迁移学习比赛](#5平安医疗科技疾病问答迁移学习比赛)
    * [6.天池“公益AI之星”挑战赛--新冠疫情相似句对判定大赛](#6天池公益ai之星挑战赛--新冠疫情相似句对判定大赛)
  * [中文医疗知识图谱](#中文医学知识图谱)
    * [1.CmeKG](#1cmekg)
  * [开源工具](#开源工具)
    * [分词工具](#分词工具)
      * [PKUSEG](#pkuseg)
  * [友情链接](#友情链接)
* [Medical_Natural_Language_Processing_Papers](#medical_natural_language_processing_papers)
  * [1.AAAI 2021](#1naacl-2021)
  * [2.AAAI 2021](#2aaai-2021)
  * [3.AAAI 2020](#3aaai-2020)
  * [4.ACL 2020](#4acl-2020)
  * [5.EMNLP 2020](#5emnlp-2020)


# Chinese_Medical_Natural_Language_Processing_Resources

## 中文医疗数据集

### 1.Yidu-S4K：医渡云结构化4K数据集

> Yidu-S4K 数据集源自CCKS 2019 评测任务一，即“面向中文电子病历的命名实体识别”的数据集，包括两个子任务：

> 1）医疗命名实体识别：由于国内没有公开可获得的面向中文电子病历医疗实体识别数据集，本年度保留了医疗命名实体识别任务，对2017年度数据集做了修订，并随任务一同发布。本子任务的数据集包括训练集和测试集。

> 2）医疗实体及属性抽取（跨院迁移）：在医疗实体识别的基础上，对预定义实体属性进行抽取。本任务为迁移学习任务，即在只提供目标场景少量标注数据的情况下，通过其他场景的标注数据及非标注数据进行目标场景的识别任务。本子任务的数据集包括训练集（非目标场景和目标场景的标注数据、各个场景的非标注数据）和测试集（目标场景的标注数据）。

数据集地址：[http://openkg.cn/dataset/yidu-s4k](http://openkg.cn/dataset/yidu-s4k)

### 2.Yidu-N7K：医渡云标准化7K数据集

> 数据描述：Yidu-N4K 数据集源自CHIP 2019 评测任务一，即“临床术语标准化任务”的数据集。

> 临床术语标准化任务是医学统计中不可或缺的一项任务。临床上，关于同一种诊断、手术、药品、检查、化验、症状等往往会有成百上千种不同的写法。标准化（归一）要解决的问题就是为临床上各种不同说法找到对应的标准说法。有了术语标准化的基础，研究人员才可对电子病历进行后续的统计分析。本质上，临床术语标准化任务也是语义相似度匹配任务的一种。但是由于原词表述方式过于多样，单一的匹配模型很难获得很好的效果。

数据集地址：[http://openkg.cn/dataset/yidu-n7k](http://openkg.cn/dataset/yidu-n7k)

### 3.瑞金医院MMC人工智能辅助构建知识图谱大赛

> 赛题描述：本次大赛旨在通过糖尿病相关的教科书、研究论文来做糖尿病文献挖掘并构建糖尿病知识图谱。参赛选手需要设计高准确率，高效的算法来挑战这一科学难题。第一赛季课题为“基于糖尿病临床指南和研究论文的实体标注构建”，第二赛季课题为“基于糖尿病临床指南和研究论文的实体间关系构建”。

数据集地址：[https://tianchi.aliyun.com/competition/entrance/231687/information](https://tianchi.aliyun.com/competition/entrance/231687/information)

### 4.中文医药方面的问答数据集

> 数据描述：该数据集由IEEE中一篇论文中提出，名为：Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection，他是一个面向中文医疗方向的问答数据集，数量级别达10万级。
> 文件说明：questions.csv：所有的问题及其内容；answers.csv：所有问题的答案；train_candidates.txt， dev_candidates.txt， test_candidates.txt：将上述两个文件进行了拆分。

数据集地址：[https://github.com/zhangsheng93/cMedQA2](https://github.com/zhangsheng93/cMedQA2)

### 5.平安医疗科技疾病问答迁移学习比赛

> 任务描述：本次比赛是chip2019中的评测任务二，由平安医疗科技主办。本次评测任务的主要目标是针对中文的疾病问答数据，进行病种间的迁移学习。具体而言，给定来自5个不同病种的问句对，要求判定两个句子语义是否相同或者相近。所有语料来自互联网上患者真实的问题，并经过了筛选和人工的意图匹配标注。首页说明了相关数据的格式。

数据集地址：[https://www.biendata.xyz/competition/chip2019/](https://www.biendata.xyz/competition/chip2019/) 需注册才能下载

### 6.天池“公益AI之星”挑战赛--新冠疫情相似句对判定大赛

> 赛制说明：比赛主打疫情相关的呼吸领域的真实数据积累，数据粒度更加细化，判定难度相比多科室文本相似度匹配更高，同时问答数据也更具时效性。本着宁缺毋滥的原则，问题的场地限制在20字以内，形成相对规范的句对。要求选手通过自然语义算法和医学知识识别相似问答和无关的问题。相关数据说明参见比赛网址首页。

数据集地址：[https://tianchi.aliyun.com/competition/entrance/231776/information](https://tianchi.aliyun.com/competition/entrance/231776/information) 需注册才能下载


## 中文医学知识图谱

### 1.CMEKG

> 知识图谱简介：CMeKG（Chinese Medical Knowledge Graph）是利用自然语言处理与文本挖掘技术，基于大规模医学文本数据，以人机结合的方式研发的中文医学知识图谱。CMeKG的构建参考了ICD、ATC、SNOMED、MeSH等权威的国际医学标准以及规模庞大、多源异构的临床指南、行业标准、诊疗规范与医学百科等医学文本信息。CMeKG 1.0包括：6310种疾病、19853种药物（西药、中成药、中草药）、1237种诊疗技术及设备的结构化知识描述，涵盖疾病的临床症状、发病部位、药物治疗、手术治疗、鉴别诊断、影像学检查、高危因素、传播途径、多发群体、就诊科室等以及药物的成分、适应症、用法用量、有效期、禁忌证等30余种常见关系类型，CMeKG描述的概念关系实例及属性三元组达100余万。

CMEKG图谱地址：[http://cmekg.pcl.ac.cn/](http://cmekg.pcl.ac.cn/)

## 开源工具

### 分词工具

#### PKUSEG

pkuseg 是由北京大学推出的基于论文PKUSEG: A Toolkit for Multi-Domain Chinese Word Segmentation 的工具包。其简单易用，支持细分领域分词，有效提升了分词准确度。

> pkuseg具有如下几个特点：
> 1.多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 我们目前支持了新闻领域，网络领域，医药领域，旅游领域，以及混合领域的分词预训练模型。在使用中，如果用户明确待分词的领域，可加载对应的模型进行分词。如果用户无法确定具体领域，推荐使用在混合领域上训练的通用模型。各领域分词样例可参考 example.txt。
> 2.更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。
> 3.支持用户自训练模型。支持用户使用全新的标注数据进行训练。
> 4.支持词性标注。

项目地址：[https://github.com/lancopku/pkuseg-python](https://github.com/lancopku/pkuseg-python)

## 友情链接

[awesome_Chinese_medical_NLP](https://github.com/GanjinZero/awesome_Chinese_medical_NLP)
[Chinese_medical_NLP](https://github.com/lrs1353281004/Chinese_medical_NLP)



# Medical_Natural_Language_Processing_Papers

医学自然语言处理相关论文汇总，目前主要汇总了EMNLP2020、ACL2020和COLING2020, NAACL2020、EMNLP2021和ACL2021等相关会议还未公布榜单，公布榜单之后会持续更新。

## 1.NAACL 2021
### 本体

The Biomaterials Annotator: a system for ontology-based concept annotation of biomaterials text

论文地址：[https://aclanthology.org/2021.sdp-1.5/](https://aclanthology.org/2021.sdp-1.5/)
### 疾病分类

Towards BERT-based Automatic ICD Coding: Limitations and Opportunities

论文地址：[https://aclanthology.org/2021.bionlp-1.6/](https://aclanthology.org/2021.bionlp-1.6/)

### 小样本学习

Scalable Few-Shot Learning of Robust Biomedical Name Representations

论文地址：[https://aclanthology.org/2021.bionlp-1.3/](https://aclanthology.org/2021.bionlp-1.3/)

### 规范化

Triplet-Trained Vector Space and Sieve-Based Search Improve Biomedical Concept Normalization

论文地址：[https://aclanthology.org/2021.bionlp-1.2/](https://aclanthology.org/2021.bionlp-1.2/)

### 预训练模型

UmlsBERT: Clinical Domain Knowledge Augmentation of Contextual Embeddings Using the Unified Medical Language System Metathesaurus

论文地址：[https://aclanthology.org/2021.naacl-main.139/](https://aclanthology.org/2021.naacl-main.139/)

Self-Alignment Pretraining for Biomedical Entity Representations

论文地址：[https://aclanthology.org/2021.naacl-main.334/](https://aclanthology.org/2021.naacl-main.334/)

Are we there yet? Exploring clinical domain knowledge of BERT models

论文地址：[https://aclanthology.org/2021.bionlp-1.5/](https://aclanthology.org/2021.bionlp-1.5/)

Stress Test Evaluation of Biomedical Word Embeddings

论文地址：[https://aclanthology.org/2021.bionlp-1.13/](https://aclanthology.org/2021.bionlp-1.13/)

BioELECTRA:Pretrained Biomedical text Encoder using Discriminators

论文地址：[https://aclanthology.org/2021.bionlp-1.16/](https://aclanthology.org/2021.bionlp-1.16/)

Improving Biomedical Pretrained Language Models with Knowledge

论文地址：[https://aclanthology.org/2021.bionlp-1.20/](https://aclanthology.org/2021.bionlp-1.20/)

EntityBERT: Entity-centric Masking Strategy for Model Pretraining for the Clinical Domain

论文地址：[https://aclanthology.org/2021.bionlp-1.21/](https://aclanthology.org/2021.bionlp-1.21/)

ChicHealth @ MEDIQA 2021: Exploring the limits of pre-trained seq2seq models for medical summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.29/](https://aclanthology.org/2021.bionlp-1.29/)

### 命名实体识别

Exploring Word Segmentation and Medical Concept Recognition for Chinese Medical Texts

论文地址：[https://aclanthology.org/2021.bionlp-1.23/](https://aclanthology.org/2021.bionlp-1.23/)

### 因果关系

Are we there yet? Exploring clinical domain knowledge of BERT models

论文地址：[https://aclanthology.org/2021.bionlp-1.5/](https://aclanthology.org/2021.bionlp-1.5/)

### 关系抽取

Improving BERT Model Using Contrastive Learning for Biomedical Relation Extraction

论文地址：[https://aclanthology.org/2021.bionlp-1.1/](https://aclanthology.org/2021.bionlp-1.1/)

### 实体链接

Clustering-based Inference for Biomedical Entity Linking

论文地址：[https://aclanthology.org/2021.naacl-main.205/](https://aclanthology.org/2021.naacl-main.205/)

End-to-end Biomedical Entity Linking with Span-based Dictionary Matching

论文地址：[https://aclanthology.org/2021.bionlp-1.18/](https://aclanthology.org/2021.bionlp-1.18/)

Word-Level Alignment of Paper Documents with their Electronic Full-Text Counterparts

论文地址：[https://aclanthology.org/2021.bionlp-1.19/](https://aclanthology.org/2021.bionlp-1.19/)

### 语言模型

BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA

论文地址：[https://aclanthology.org/2021.bionlp-1.24/](https://aclanthology.org/2021.bionlp-1.24/)

Semi-Supervised Language Models for Identification of Personal Health Experiential from Twitter Data: A Case for Medication Effects

论文地址：[https://aclanthology.org/2021.bionlp-1.25/](https://aclanthology.org/2021.bionlp-1.25/)

Assertion Detection in Clinical Notes: Medical Language Models to the Rescue?

论文地址：[https://aclanthology.org/2021.nlpmc-1.5/](https://aclanthology.org/2021.nlpmc-1.5/)

### 摘要生成

UETrice at MEDIQA 2021: A Prosper-thy-neighbour Extractive Multi-document Summarization Model

论文地址：[https://aclanthology.org/2021.bionlp-1.36/](https://aclanthology.org/2021.bionlp-1.36/)

IBMResearch at MEDIQA 2021: Toward Improving Factual Correctness of Radiology Report Abstractive Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.35/](https://aclanthology.org/2021.bionlp-1.35/)

Optum at MEDIQA 2021: Abstractive Summarization of Radiology Reports using simple BART Finetuning

论文地址：[https://aclanthology.org/2021.bionlp-1.32/](https://aclanthology.org/2021.bionlp-1.32/)

MNLP at MEDIQA 2021: Fine-Tuning PEGASUS for Consumer Health Question Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.37/](https://aclanthology.org/2021.bionlp-1.37/)

UETfishes at MEDIQA 2021: Standing-on-the-Shoulders-of-Giants Model for Abstractive Multi-answer Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.38/](https://aclanthology.org/2021.bionlp-1.38/)

Towards Automating Medical Scribing : Clinic Visit Dialogue2Note Sentence Alignment and Snippet Summarization

论文地址：[https://aclanthology.org/2021.nlpmc-1.2/](https://aclanthology.org/2021.nlpmc-1.2/)

paht_nlp @ MEDIQA 2021: Multi-grained Query Focused Multi-Answer Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.10/](https://aclanthology.org/2021.bionlp-1.10/)

### 事件抽取

Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network

论文地址：[https://aclanthology.org/2021.naacl-main.156/](https://aclanthology.org/2021.naacl-main.156/)

### 迁移学习

UCSD-Adobe at MEDIQA 2021: Transfer Learning and Answer Sentence Selection for Medical Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.28/](https://aclanthology.org/2021.bionlp-1.28/)

SB_NITK at MEDIQA 2021: Leveraging Transfer Learning for Question Summarization in Medical Domain

论文地址：[https://aclanthology.org/2021.bionlp-1.31/](https://aclanthology.org/2021.bionlp-1.31/)

NLM at MEDIQA 2021: Transfer Learning-based Approaches for Consumer Question and Multi-Answer Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.34/](https://aclanthology.org/2021.bionlp-1.34/)

### 数据集

emrKBQA: A Clinical Knowledge-Base Question Answering Dataset

论文地址：[https://aclanthology.org/2021.bionlp-1.7/](https://aclanthology.org/2021.bionlp-1.7/)

### 多模态

QIAI at MEDIQA 2021: Multimodal Radiology Report Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.33/](https://aclanthology.org/2021.bionlp-1.33/)

### 对话

Overview of the MEDIQA 2021 Shared Task on Summarization in the Medical Domain

论文地址：[https://aclanthology.org/2021.bionlp-1.8/](https://aclanthology.org/2021.bionlp-1.8/)

WBI at MEDIQA 2021: Summarizing Consumer Health Questions with Generative Transformers

论文地址：[https://aclanthology.org/2021.bionlp-1.9/](https://aclanthology.org/2021.bionlp-1.9/)

Gathering Information and Engaging the User ComBot: A Task-Based, Serendipitous Dialog Model for Patient-Doctor Interactions

论文地址：[https://aclanthology.org/2021.nlpmc-1.3/](https://aclanthology.org/2021.nlpmc-1.3/)

Extracting Appointment Spans from Medical Conversations

论文地址：[https://aclanthology.org/2021.nlpmc-1.6/](https://aclanthology.org/2021.nlpmc-1.6/)

Building blocks of a task-oriented dialogue system in the healthcare domain

论文地址：[https://aclanthology.org/2021.nlpmc-1.7/](https://aclanthology.org/2021.nlpmc-1.7/)

Medically Aware GPT-3 as a Data Generator for Medical Dialogue Summarization

论文地址：[https://aclanthology.org/2021.nlpmc-1.9/](https://aclanthology.org/2021.nlpmc-1.9/)

### 文本生成

BBAEG: Towards BERT-based Biomedical Adversarial Example Generation for Text Classification

论文地址：[https://aclanthology.org/2021.naacl-main.423/](https://aclanthology.org/2021.naacl-main.423/)

### 问答

NCUEE-NLP at MEDIQA 2021: Health Question Summarization Using PEGASUS Transformers

论文地址：[https://aclanthology.org/2021.bionlp-1.30/](https://aclanthology.org/2021.bionlp-1.30/)

damo_nlp at MEDIQA 2021: Knowledge-based Preprocessing and Coverage-oriented Reranking for Medical Question Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.12/](https://aclanthology.org/2021.bionlp-1.12/)

### 表示学习

Word centrality constrained representation for keyphrase extraction

论文地址：[https://aclanthology.org/2021.bionlp-1.17/](https://aclanthology.org/2021.bionlp-1.17/)

### Others

Contextual explanation rules for neural clinical classifiers

论文地址：[https://aclanthology.org/2021.bionlp-1.22/](https://aclanthology.org/2021.bionlp-1.22/)

Context-aware query design combines knowledge and data for efficient reading and reasoning

论文地址：[https://aclanthology.org/2021.bionlp-1.26/](https://aclanthology.org/2021.bionlp-1.26/)

Measuring the relative importance of full text sections for information retrieval from scientific literature.

论文地址：[https://aclanthology.org/2021.bionlp-1.27/](https://aclanthology.org/2021.bionlp-1.27/)

Automatic Speech-Based Checklist for Medical Simulations

论文地址：[https://aclanthology.org/2021.nlpmc-1.4/](https://aclanthology.org/2021.nlpmc-1.4/)

Joint Summarization-Entailment Optimization for Consumer Health Question Understanding

论文地址：[https://aclanthology.org/2021.nlpmc-1.8/](https://aclanthology.org/2021.nlpmc-1.8/)

Detecting Anatomical and Functional Connectivity Relations in Biomedical Literature via Language Representation Models

论文地址：[https://aclanthology.org/2021.sdp-1.4/](https://aclanthology.org/2021.sdp-1.4/)

BDKG at MEDIQA 2021: System Report for the Radiology Report Summarization Task

论文地址：[https://aclanthology.org/2021.bionlp-1.11/](https://aclanthology.org/2021.bionlp-1.11/)

## 2.AAAI 2021

Subtype-Aware Unsupervised Domain Adaptation for Medical Diagnosis

论文地址：[https://arxiv.org/pdf/2101.00318.pdf](https://arxiv.org/pdf/2101.00318.pdf)


Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation

论文地址：[https://arxiv.org/pdf/2012.11988.pdf](https://arxiv.org/pdf/2012.11988.pdf)


A Lightweight Neural Model for Biomedical Entity Linking

论文地址：[https://arxiv.org/pdf/2012.08844.pdf](https://arxiv.org/pdf/2012.08844.pdf)


Automated Lay Language Summarization of Biomedical Scientific Reviews

论文地址：[https://arxiv.org/pdf/2012.12573.pdf](https://arxiv.org/pdf/2012.12573.pdf)


MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization

论文地址：[https://arxiv.org/pdf/1902.10118.pdf](https://arxiv.org/pdf/1902.10118.pdf)


MELINDA: A Multimodal Dataset for Biomedical Experiment Method Classification

论文地址：[https://arxiv.org/pdf/2012.09216.pdf](https://arxiv.org/pdf/2012.09216.pdf)


## 3.AAAI 2020

Simultaneously Linking Entities and Extracting Relations from Biomedical Text without Mention-Level Supervision

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6236](https://aaai.org/ojs/index.php/AAAI/article/view/6236)


Can Embeddings Adequately Represent Medical Terminology? New Large-Scale Medical Term Similarity Datasets Have the Answer!

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6404](https://aaai.org/ojs/index.php/AAAI/article/view/6404)


Understanding Medical Conversations with Scattered Keyword Attention and Weak Supervision from Responses

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6412](https://aaai.org/ojs/index.php/AAAI/article/view/6412)


Learning Conceptual-Contextual Embeddings for Medical Text

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6504](https://aaai.org/ojs/index.php/AAAI/article/view/6504)


LATTE: Latent Type Modeling for Biomedical Entity Linking

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6526](https://aaai.org/ojs/index.php/AAAI/article/view/6526)


## 4.ACL 2020



## 5.EMNLP 2020

Infusing Disease Knowledge into BERT for Health Question Answering, Medical Inference and Disease Name Recognition
 
论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.372/](https://www.aclweb.org/anthology/2020.emnlp-main.372/)


### 机器翻译

Evaluation of Machine Translation Methods applied to Medical Terminologies

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.7/](https://www.aclweb.org/anthology/2020.louhi-1.7/)


A Multilingual Neural Machine Translation Model for Biomedical Data

论文地址：[https://www.aclweb.org/anthology/2020.nlpcovid19-2.16/](https://www.aclweb.org/anthology/2020.nlpcovid19-2.16/)


Findings of the WMT 2020 Biomedical Translation Shared Task: Basque, Italian and Russian as New Additional Languages

论文地址：[https://www.aclweb.org/anthology/2020.wmt-1.76/](https://www.aclweb.org/anthology/2020.wmt-1.76/)


Elhuyar submission to the Biomedical Translation Task 2020 on terminology and abstracts translation

论文地址：[https://www.aclweb.org/anthology/2020.wmt-1.87/](https://www.aclweb.org/anthology/2020.wmt-1.87/)


Pretrained Language Models and Backtranslation for English-Basque Biomedical Neural Machine Translation

论文地址：[https://www.aclweb.org/anthology/2020.wmt-1.89/](https://www.aclweb.org/anthology/2020.wmt-1.89/)



### 机器阅读理解

Towards Medical Machine Reading Comprehension with Structural Knowledge and Plain Text

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.111/](https://www.aclweb.org/anthology/2020.emnlp-main.111/)



### 实体规范化

A Knowledge-driven Generative Model for Multi-implication Chinese Medical Procedure Entity Normalization

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.116/](https://www.aclweb.org/anthology/2020.emnlp-main.116/)


Target Concept Guided Medical Concept Normalization in Noisy User-Generated Texts

论文地址：[https://www.aclweb.org/anthology/2020.deelio-1.8/](https://www.aclweb.org/anthology/2020.deelio-1.8/)


Medical Concept Normalization in User-Generated Texts by Learning Target Concept Embeddings

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.3/](https://www.aclweb.org/anthology/2020.louhi-1.3/)



### 命名实体识别

Assessment of DistilBERT performance on Named Entity Recognition task for the detection of Protected Health Information and medical concepts

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.18/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.18/)



### 关系抽取

FedED: Federated Learning via Ensemble Distillation for Medical Relation Extraction

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.165/](https://www.aclweb.org/anthology/2020.emnlp-main.165/)



### 实体链接

COMETA: A Corpus for Medical Entity Linking in the Social Media

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.253/](https://www.aclweb.org/anthology/2020.emnlp-main.253/)



Simple Hierarchical Multi-Task Neural End-To-End Entity Linking for Biomedical Text

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.2/](https://www.aclweb.org/anthology/2020.louhi-1.2/)



### 语言模型

BioMegatron: Larger Biomedical Domain Language Model

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.379/](https://www.aclweb.org/anthology/2020.emnlp-main.379/)


Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.17/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.17/)


Inexpensive Domain Adaptation of Pretrained Language Models: Case Studies on Biomedical NER and Covid-19 QA

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.134/](https://www.aclweb.org/anthology/2020.findings-emnlp.134/)


On the effectiveness of small, discriminatively pre-trained language representation models for biomedical text mining

论文地址：[https://www.aclweb.org/anthology/2020.sdp-1.12/](https://www.aclweb.org/anthology/2020.sdp-1.12/)



### 事件抽取

Biomedical Event Extraction as Sequence Labeling

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.431/](https://www.aclweb.org/anthology/2020.emnlp-main.431/)


Biomedical Event Extraction with Hierarchical Knowledge Graphs

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.114/](https://www.aclweb.org/anthology/2020.findings-emnlp.114/)



### 数据集

COMETA: A Corpus for Medical Entity Linking in the Social Media

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.253/](https://www.aclweb.org/anthology/2020.emnlp-main.253/)


MedDialog: Large-scale Medical Dialogue Datasets

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.743/](https://www.aclweb.org/anthology/2020.emnlp-main.743/)


MeDAL: Medical Abbreviation Disambiguation Dataset for Natural Language Understanding Pretraining

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.15/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.15/)


MedICaT: A Dataset of Medical Images, Captions, and Textual References

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.191/](https://www.aclweb.org/anthology/2020.findings-emnlp.191/)


GGPONC: A Corpus of German Medical Text with Rich Metadata Based on Clinical Practice Guidelines

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.5/](https://www.aclweb.org/anthology/2020.louhi-1.5/)



### 基于国外临床医学数据的NLP研究

Information Extraction from Swedish Medical Prescriptions with Sig-Transformer Encoder

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.5/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.5/)


Classification of Syncope Cases in Norwegian Medical Records

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.9/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.9/)



### 对话

Weakly Supervised Medication Regimen Extraction from Medical Conversations

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.20/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.20/)


Dr. Summarize: Global Summarization of Medical Dialogue by Exploiting Local Structures.

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.335/](https://www.aclweb.org/anthology/2020.findings-emnlp.335/)



### 文本生成

Reinforcement Learning with Imbalanced Dataset for Data-to-Text Medical Report Generation

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.202/](https://www.aclweb.org/anthology/2020.findings-emnlp.202/)


Generating Accurate Electronic Health Assessment from Medical Graph

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.336/](https://www.aclweb.org/anthology/2020.findings-emnlp.336/)



### 问答

Biomedical Event Extraction as Multi-turn Question Answering

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.10/](https://www.aclweb.org/anthology/2020.louhi-1.10/)



### 推荐

COVID-19: A Semantic-Based Pipeline for Recommending Biomedical Entities

论文地址：[https://www.aclweb.org/anthology/2020.nlpcovid19-2.20/](https://www.aclweb.org/anthology/2020.nlpcovid19-2.20/)



### 主题模型


Developing a Curated Topic Model for COVID-19 Medical Research Literature

论文地址：[https://www.aclweb.org/anthology/2020.nlpcovid19-2.30/](https://www.aclweb.org/anthology/2020.nlpcovid19-2.30/)



### 表示学习

ERLKG: Entity Representation Learning and Knowledge Graph based association analysis of COVID-19 through mining of unstructured biomedical corpora

论文地址：[https://www.aclweb.org/anthology/2020.sdp-1.15/](https://www.aclweb.org/anthology/2020.sdp-1.15/)


Learning Informative Representations of Biomedical Relations with Latent Variable Models

论文地址：[https://www.aclweb.org/anthology/2020.sustainlp-1.3/](https://www.aclweb.org/anthology/2020.sustainlp-1.3/)



### Others

Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.8/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.8/)


Summarizing Chinese Medical Answer with Graph Convolution Networks and Question-focused Dual Attention

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.2/](https://www.aclweb.org/anthology/2020.findings-emnlp.2/)


Sequential Span Classification with Neural Semi-Markov CRFs for Biomedical Abstracts

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.77/](https://www.aclweb.org/anthology/2020.findings-emnlp.77/)


Characterizing the Value of Information in Medical Notes

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.187/](https://www.aclweb.org/anthology/2020.findings-emnlp.187/)


Querying Across Genres for Medical Claims in News

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.139/](https://www.aclweb.org/anthology/2020.emnlp-main.139/)


An efficient representation of chronological events in medical texts

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.11/](
