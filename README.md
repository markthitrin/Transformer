# Transform From Scratch
This project is created during my internship at Prof. Kenjiro Taura Laboratory at University of Tokyo. The main focus of this projet is to compare the performance achieved of Chapel to C++, Python in training a transformer model. The Chapel and C++code is implemented from scratch and the python is obtained from [GitHub](https://github.com/markthitrin/Transformer.git)
## Branch
- **Master** : contain all versions including the cuda implementation from scratch as well, which is a side project.
- **SingleThread** : contain the single thread implementation of C++, Chapel, 2 versions of Pytorch.
- **MultiThread** : contain the multithread thread implementation of C++, Chapel, 2 versions of Pytorch.
## Build
**C++**
```bash
cd ./C++
make -j
```
**Chapel**
```bash
cd ./Chapel
chpl ./main.chpl --fast
```
**cuda**
```bash
cd ./cuda
make -j
```
**python**
For Pytorch A (orignal implementation from [GitHub](https://github.com/markthitrin/Transformer.git))
```bash
cd ./python/Transformer-from-Scratch
python ./train.py
```
For Pytorch B (orignal implementation from [GitHub](https://github.com/markthitrin/Transformer.git) with transformerBlock replaced by `torch.nn.transformer`)
```bash
cd ./python/Pytorch
python ./train.py
```
