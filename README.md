# A Privacy-Preserving Federated Framework for LLM-based Recommendation
This is the pytorch implementation of our paper
> A Privacy-Preserving Federated Framework for LLM-based Recommendation


## Environment
- Anaconda 3
- Python 3.8.13
- Pytorch 2.1.1
- Numpy 1.22.3

## RecFormer-based
Run the RecFormer-based PPLR on the Games dataset:
- download pretrained checkpoints into the ./pretrain_ckpt: [RecformerForSeqRec](https://drive.google.com/file/d/1BEboY3NxAUOBe6YwYZ_RsQ4BR6IIbl0-/view?usp=sharing)
- Assign os.environ['LD_LIBRARY_PATH'] in fintune.py
```bash
cd RecFormer_PPLR
bash fintune.sh
```

## BIGRec-based
### Training
Run the RecFormer-based PPLR on the Games dataset:
- Assign os.environ['LD_LIBRARY_PATH'] in fintune.py
- Assign LLaMA base model path in train.sh
```bash
cd BigRec_PPLR
bash train.sh
```
### Inference
- Assign LLaMA base model path in inference.py
```bash
cd BigRec_PPLR
python inference.py
```
### Evaluate
- Assign LLaMA base model path in evaluate.py
```bash
cd BigRec_PPLR/data/games
python evaluate.sh
```