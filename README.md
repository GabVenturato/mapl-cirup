# mapl-cirup
MArkov PLanning with CIRcuit bellman UPdates (pronounced as "maple syrup") is an
MDP solver which learns a symbolic policy function, performing the Bellman 
update with dynamic decision cricuits.

## Requirements
- Python 3 (tested with `>=3.8`)
- [ProbLog >=2.2.2](https://dtai.cs.kuleuven.be/problog/)
- [PySDD >=0.2.10](https://github.com/wannesm/PySDD)
- numpy >= 1.23
- graphviz (for printing circuits)
- TensorFlow 2

## Learning
To reproduce the results from the paper one can use:
```
python train.py examples_learning/coffee3_1994/coffee3_init$((41+$run)).pl \
examples_learning/coffee3_1994/dataset_n100_trajlen5_seed1338.pickle 50 10 \
$run --lr 0.1
```
with `$run` ranging from 1 to 10.

The extra results in the appendix can be obtained similarly with:

- ```
  python train.py examples_learning/coffee2_1994/coffee2_init$((41+$run)).pl \
  examples_learning/coffee2_1994/dataset_n100_trajlen5_seed1338.pickle 50 10 \
  $run --lr 0.1
  ```
- ```
  python train.py examples_learning/coffee3_1994/coffee3_init$((41+$run)).pl \
  examples_learning/coffee3_1994/dataset_n10_trajlen5_seed1337.pickle 200 5 \
  $run --lr 0.1
  ```

**Note:** We use TensorFlow to extract the gradients but the model does not run on GPU.
Therefore, you may want to specify `export CUDA_VISIBLE_DEVICES=""` before running.