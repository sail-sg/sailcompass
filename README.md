# SailCompass: Towards Reproducible and Robust Evaluation for Southeast Asian Languages

In this work, we present SailCompass, a comprehensive suite of evaluation scripts designed for robust and reproducible evaluation of multilingual language models targeting Southeast Asian languages. 

SailCompass encompasses three major SEA languages and covers eight primary tasks using 14 datasets, spanning three task types: generation, multiple-choice questions, and classification.

### Environment Setup

We use [OpenCompass](https://github.com/open-compass/opencompass) to evaluate the models. To install the required packages, run the following command under this folder:

```bash
conda create --name sailcompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate sailcompass

git clone https://github.com/sail-sg/sailcompass sailcompass

###git clone submodule
cd sailcompass
git submodule update --init --recursive

###git clone opencompass and copy the config
bash setup_environment.sh

###download eval data from huggingface
mkdir data
python download_eval_data.py

```

### Evaluation Script

To build the evaluation script, run the following command under this folder:

```bash
bash setup_sailcompass.sh
```

### Run Evaluation

To run the evaluation, run the following command under this folder:

```bash
cd opencompass
python run.py configs/eval_sailcompass.py -w outputs/sailcompass --num-gpus 1 --max-num-workers 64 --debug
```



## Acknowledgment

Thanks to the contributors of the [opencompass](https://github.com/open-compass/opencompass).


## Citing this work

If you use sailcompass benchmark in your work, please cite

```
@misc{sailcompass,
      title={SailCompass: Towards Reproducible and Robust Evaluation for Southeast Asian Languages}, 
      author={Jia Guo and Longxu Dou and Guangtao Zeng and Stanley Kok and Wei Lu and Qian Liu},
      year={2024},
}
```

## Contact

If you have any questions, please raise an issue on our GitHub repository or contact <a href="mailto:doulx@sea.com">doulx@sea.com</a>.