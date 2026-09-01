\# GPT-2 From Scratch 100M



参考 Karpathy 的 build-nanogpt，从零实现 GPT-2，并使用 FineWeb-Edu 100M Token 进行单卡训练实验。



\## 开发环境



\- 操作系统：Windows

\- Python：3.11

\- GPU：NVIDIA GeForce RTX 5080

\- CUDA：12.8

\- 训练框架：PyTorch



\## 安装环境



创建并激活 Conda 环境：



```powershell

conda create -n gpt2-from-scratch-100m python=3.11 -y

conda activate gpt2-from-scratch-100m



\##当前版本



Python: 3.11.16 | packaged by Anaconda, Inc. | (main, Aug 27 2026, 14:36:16) \[MSC v.1942 64 bit (AMD64)]

PyTorch: 2.11.0+cu128

CUDA: 12.8

GPU: NVIDIA GeForce RTX 5080 Laptop GPU



