# Transformer From Scractch (Single Thread)
You are currently on the single thread branch. To see the whole project please visit master branch. This branch contain the single thread imlpementation of C++, Chapel, both 2 version of the Python.
## Build and Run
Every models configuration is set in the config file. The architecture is based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The configuration parameter are the following for every model.
- dModel :  the number of features in the encoder/decoder inputs 
- sequenceLength : the maximum sequence length of both input and ouput
- dFF : the dimension of the feedforward network model
- N : the number of sub-layers in the encoder and decoder
- head : number of heads in multihea attention layer
- srcVocab : number of token in the source language.
- tgtVocab : number of token in the target language.
- batch_size : size of batch
- dropout : probability of dropping out in dropout later.
To build each model
- trainingIteration : number of iteration to traing on the problem.
### C++
0. Please make sure clang is avaliable on your machine
1. change working directory to C++
```bash
cd ./C++
```
2. configure the transformer in the `/Config.h` file
3. configure the number `traniningIteration` which is used to compute the time spent on each layer during the training of the model, and `testingIteration`.
4. build the model using `clang` through `make` command. This will compile the cpp files and store object file in ./build
```bash
make clean
make -j
```
5. run the compiled code
```bash
./main
```

### Chapel
0. Please make sure Chapel compiler is avaliable on your machine.
1. change working directory to Chapel
```bash
cd ./Chapel
```
2. configure the transformer in the `./Config.chpl` file
3. configure the number `traniningIteration` which is used to compute the time spent on each layer during the training of the model, and `testingIteration`.
4. build the model
```bash
chpl ./main.chpl --fast
```
5. run the compiled code with setting `CHPL_RT_NUM_THREADS_PER_LOCALE` to force `randomstream` in dropout layer to work on single thread
```
CHPL_RT_NUM_THREADS_PER_LOCALE=1 ./main
```

### PyTorch A and PyTorchB
1. Change working directory to PyTorchA or PyTorchB
```bash
cd ./PyTorchA
```
2. configure the transformer in the `./config.py` file
3. configure the number `traniningIteration` which is used to compute the time spent on each layer during the training of the model, and `testingIteration`.
4. run the model
```bash
python ./train.py
```

## Result
The program should continuely train the model and consistently output the current iteration number. After the model complete its should be list of time of each layer in the model following with model testing result. You can view my running result at [Sheet](https://docs.google.com/spreadsheets/d/1aHkE9Ckl0-waxVwu-f4dIJ0peM6jIUQv3IU1-bFa0p0/edit?usp=sharing)

