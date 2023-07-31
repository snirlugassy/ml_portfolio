# ML In Portfolio Selection - Final Project

### Requirements

Installing PyTorch:

OSX
```bash
python3 -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

Linux
```bash
python3 -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Installing other requirements:
```bash
python3 -m pip instal -r requirements.txt
```
Algorithms to test:

MVP

Anna
Linear regression / with second order
Decision trees / Random forest
LARS / LASSO regression 

Eden
Online learning (follow the leader…)
Reinforcement learning
1D-CNN (torch.nn.Conv1d)

Snir
XGboost (boosted trees)
SVM
MLP (network with only linear + activations)
RNN (LSTM, GRU, …) https://arxiv.org/pdf/2005.13665v3.pdf

Ensemble of several models
