# QR-DQN-Pytorch
这是一个**QR-DQN的Pytorch实现**。

## 依赖
```bash
gymnasium==0.29.1
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

## 如何使用代码
### 从头开始训练:
```bash
python main.py # 在CartPole-v1上训练QR-DQN
```

### 更改环境:
如果你想在不同的环境中训练，只需运行
```bash
python main.py --EnvIdex 1 # 在LunarLander-v2上训练QR-DQN
```
`--EnvIdex` 可以设置为0和1，其中
```bash
'--EnvIdex 0' 对应 'CartPole-v1'
'--EnvIdex 1' 对应 'LunarLander-v2'
```

### 使用预训练模型:
```bash
python main.py --EnvIdex 0 --render True --Loadmodel True --ModelIdex 100 # 使用预训练的QR-DQN在CartPole-v1上运行
```
```bash
python main.py --EnvIdex 1 --render True --Loadmodel True --ModelIdex 400 # 使用预训练的QR-DQN在LunarLander-v2上运行
```

### 可视化训练曲线
你可以使用[tensorboard](https://pytorch.org/docs/stable/tensorboard.html)来记录和可视化训练曲线。

- 安装 (请确保已经安装了Pytorch):
```bash
pip install tensorboard
pip install packaging
```
- 记录 (训练曲线将保存在 '**\runs**' 文件夹中):
```bash
python main.py --write True
```

- 可视化:
```bash
tensorboard --logdir runs
```

### 超参数设置
有关超参数设置的更多详细信息，请查看 'main.py'

### 参考文献
QR-DQN: Dabney W, Ostrovski G, Silver D, et al. Distributional reinforcement learning with quantile regression[J]. arXiv preprint arXiv:1710.10044, 2017.
