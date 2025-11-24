<h1>CVG_Imbalance </h1>

Official implementation for TNNLS 2025 paper: "**[On Imbalance in Case Types: Evaluating and Enhancing PLMs for Criminal Court View Generation](xxx)**"



 <div>
    <a href="https://yuquanle.github.io/CVG_Imbalance-homepage/"><img src="https://img.shields.io/badge/Homepage-CVG_Imbalance-pink"/></a> <a href="https://www.sciencedirect.com/science/article/pii/S0957417423026052">
  <img src="https://img.shields.io/badge/IEEE-TNNLS-blue?style=flat-square&logo=ieee&logoColor=white" alt="IEEE TNNLS">

</a>

   </div>

<h5> If you like this work, please give us a star ⭐ on GitHub.  </h2>


<h1>Introduction</h1> 
</div>

 <br>

</h5>
</p> 
<p align="center">
    <img src="figs/teaserFig.png"/>
<p>
    <p align="justify"> Criminal Court View Generation (CCVG) task aims
 to produce succinct and coherent summaries of fact descrip
tions, providing interpretable opinions for verdicts. Traditional
 text generation evaluation metrics, such as ROUGE, BLEU,
 and BERTSCORE, are extensively employed for this task and
 measure performance by averaging the assessment scores of
 all samples within the test set. However, these sample-averaged
 metrics encounter two primarily dilemmas: 1) they fail to fairly
 assess overall evaluation scores across different case types, and
 1) they overlook the measurement of the degree of performance
 imbalance between case types. To fill this research gap, we
 propose two novel case-type-oriented evaluation metrics: Case
type-oriented Text Generation (CTG) and Case-type-oriented
 Imbalance Performance (CIP). First, CTG mitigates the unfair
 assessment among different case types by assigning equal weight
 to each type. Second, CIP evaluates performance imbalance by
 measuring the distance between the performance of each case
 type and the overall performance. We provide three Theorems
 to elucidate the properties of CIP, demonstrating that CIP can
 effectively identify the extent to which a CCVG model achieves
 balanced generation performance across different case types.
 Furthermore, we propose an embarrassingly simple and effective
 Charge-Guided Encoder-Decoder (CGED) framework to enhance
 performance fairly across different case types in encoder-decoder
 pre-trained language models. </p>

## Code Structure
```
CVG_Imbalance/
├── dataset.py
├── evaluate_metrics_macro.py
├── evaluate_metrics.py
├── main_criminal_cvg.py
├── model.py
├── README.md
├── scripts
│   ├── evaluate_criminal_cvg.sh
│   ├── evaluate_macro_criminal_cvg.sh
│   ├── test_criminal_cvg.sh
│   └── train_criminal_cvg.sh
├── train_criminal_cvg.py
└── utils.py
```

## Datasets
We use the following datasets:
- [CJO](https://github.com/bigdata-ustc/C3VG)


## Quick Start

### Traditional text generation evaluation metrics.
```shell
git clone https://github.com/CVG_Imbalance.git
cd CVG_Imbalance/scripts
./evaluate_criminal_cvg.sh.sh
```

### CTG metrics.
<p align="center">
    <img src="figs/CTG.png"/>
<p>

```shell
./evaluate_macro_criminal_cvg.sh.sh
```

### CIP metrics.
<p align="center">
    <img src="figs/CIP.png"/>
<p>

CIP evaluates performance imbalance by measuring the distance between the performance of each case type and the overall performance. We find that CIP is equivalent to the subtraction between the traditional text generation evaluation metrics (e.g., ROUGE-L) and CTG metrics (e.g., CTG@ROUGE-L).

### CGED framework.

#### Training.
```shell
./train_criminal_cvg.sh
```

#### Infering.
```shell
./test_criminal_cvg.sh
```

## Acknowledgement
Thanks to the open-source projects ([C3VG](https://github.com/bigdata-ustc/C3VG)) for their works.

## Citation
If you find this work useful, please consider starring 🌟 this repo and citing 📑 our paper:


```bibtex
@article{le2025cvg,
  author={Le, Yuquan and Xiao, Zheng and Ding, Yan and Chng, Eng Siong and Li, Kenli},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={On Imbalance in Case Types: Evaluating and Enhancing PLMs for Criminal Court View Generation}, 
  year={2025},
  pages={1-13},
  doi={10.1109/TNNLS.2025.3622490}}
```
