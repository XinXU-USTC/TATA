

<h1 align="center">
     <br>TATA: Teaching LLMs According to Their Aptitude
<p align="center">
    <a href="https://arxiv.org/abs/2502.12022">
        <img alt="Static Badge" src="https://img.shields.io/badge/Paper-Arxiv-red">
    </a>
    <a href="">
        <img alt="Static Badge" src="https://img.shields.io/badge/HFDataset-TATA-yellow">
    </a>
</p>


## 🔥News
- *2025-02-25*: We have released our codes for data selection and evaluation.
- *2025-02-17*: We have released our paper on Arxiv.

------
**TATA** (**T**eaching LLMs **A**ccording to **T**heir **A**ptitude) is an adaptive framework that enables LLMs to personalize their reasoning strategy (CoT or TIR) spontaneously,aligning it with their intrinsic aptitude.
The overview of our TATA is depicted as follows:


## 🚀Quick Start 
For installation, you can use the following commands to set up your environment (see also [Dart-Math](https://github.com/hkust-nlp/dart-math)).

```bash
## clone our repo
git clone https://github.com/XinXU-USTC/TATA.git
cd TATA
## install dart-math environment
git clone https://github.com/hkust-nlp/dart-math.git && cd dart-math
conda create --name tata --yes python=3.11
conda activate tata
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install pebble timeout_decorator
cd ..
```

## 🗂 Data Selection
For data selection using TATA, use the following script:
```bash
cd src
bash scripts/get_score.sh
```

## 🔨 Training
Please refer to [Dart-Math](https://github.com/hkust-nlp/dart-math). In fact, any open-source SFT repo can be used, e.g., [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [LMFlow](https://github.com/OptimalScale/LMFlow).


## ⚖️ Evaluation

For evaluation, you can use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer.inference \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --data_name ${DATA_NAME} \
  --split ${SPLIT} \
  --prompt_type ${PROMPT_TYPE} \
  --num_test_sample -1 \
  --seed 0 \
  --temperature 0 \
  --n_sampling 1 \
  --top_p 1 \
  --start 0 \
  --end -1
```
or simply use the following bash script:
```bash
cd src
bash scripts/infer.sh
```


## 💬 Citation
Thanks for the open-source code of [ToRA](https://github.com/microsoft/ToRA) and [Dart-Math](https://github.com/hkust-nlp/dart-math).


If you find our work interesting and meaningful, welcome to give a 🌟 to our repo and cite our paper.
```
@article{xu2025tata,
  title={Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving},
  author={Xu, Xin and Xu, Yan and Chen, Tianhao and Yan, Yuchen and Liu, Chengwu and Chen, Zaoyu and Wang, Yufei and Yin, Yichun and Wang, Yasheng and Shang, Lifeng and others},
  journal={arXiv preprint arXiv:2502.12022},
  year={2025}
}
```

