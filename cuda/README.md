# Build
To compile and get the executable, run the following on this directory.
```bash
mkdir build
make -j$(nproc)
```
You will get one executable program name "BitcoinMiner" in this directory.

# Test
I have done some test on the model to ensure the correctness of the model.
The test code is in the **./Test** directory which only works with previous commit
which each layer contain testing code. The test ensure the forward and backward 
operation of the model using the generated test cases from **../python/Testcase**.
The result can be found in **./TestResult**

However, the result contain some computational error which might occur from using fast math operation
provided by CUDA in the cuda kernel. Another thing that should be concern is that the test cover
up until the Forward, Backward and Parameters changes of the whole transformer model, and not yet to cover
loss function and prediction process.

# Limitation
This model have not been tuned to achieved perfect performance. There are many optimization that could be done, including the followings
- Tune the block size for each individual kernel.
- Combine kernels to exploit the locality.
- Pin priority nodes.
- Draw better dependency graph, as currently, each layer has to finish before continue to the next layer.
- Hint L2 keep memory.
- Make the training process completely on device.
