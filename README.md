# Chinese_Medical_Natural_Language_Processing

## 中文医疗数据集

### Yidu-S4K：医渡云结构化4K数据集

数据集地址：[http://openkg.cn/dataset/yidu-s4k](http://openkg.cn/dataset/yidu-s4k)

### Yidu-N7K：医渡云标准化7K数据集

数据集地址：[http://openkg.cn/dataset/yidu-n7k](http://openkg.cn/dataset/yidu-n7k)

### 瑞金医院MMC人工智能辅助构建知识图谱大赛

数据集地址：[https://tianchi.aliyun.com/competition/entrance/231687/information](https://tianchi.aliyun.com/competition/entrance/231687/information)

## 中文医学知识图谱

### CMEKG

> 知识图谱简介：CMeKG（Chinese Medical Knowledge Graph）是利用自然语言处理与文本挖掘技术，基于大规模医学文本数据，以人机结合的方式研发的中文医学知识图谱。CMeKG的构建参考了ICD、ATC、SNOMED、MeSH等权威的国际医学标准以及规模庞大、多源异构的临床指南、行业标准、诊疗规范与医学百科等医学文本信息。CMeKG 1.0包括：6310种疾病、19853种药物（西药、中成药、中草药）、1237种诊疗技术及设备的结构化知识描述，涵盖疾病的临床症状、发病部位、药物治疗、手术治疗、鉴别诊断、影像学检查、高危因素、传播途径、多发群体、就诊科室等以及药物的成分、适应症、用法用量、有效期、禁忌证等30余种常见关系类型，CMeKG描述的概念关系实例及属性三元组达100余万。

CMEKG图谱地址：[http://cmekg.pcl.ac.cn/](http://cmekg.pcl.ac.cn/)



# Medical_Natural_Language_Processing_Papers

医学自然语言处理相关论文汇总，目前主要汇总了EMNLP2020、ACL2020和COLING2020,NAACL2020、EMNLP2021和ACL2021还未公布榜单，公布榜单之后会持续更新

## EMNLP 2020

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
