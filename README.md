# Dynamical Mechanism of Sampling-Based Probabilistic Inference
This repository is the code for the research of [Dynamical Mechanism of 
Sampling-Based Probabilistic Inference under Probabilistic Population Codes](https://direct.mit.edu/neco/article/doi/10.1162/neco_a_01477/109083/Dynamical-Mechanism-of-Sampling-Based).

arxiv: https://arxiv.org/abs/2106.05591

## Training
The neural network can be trained for the following two tasks.

- Cue Combination task
- Coordinate Transformation task

For an overview of the tasks, see [this paper](https://www.nature.com/articles/s41467-017-00181-8).

After setting the config file, run the train script as follows.

```bash
$ python train_cue_combination.py cfg/sample.yml
```

If you run `train_cue_combination_point.py`, your model train the cue combination task as the point estimation task.


## Inference 
Please see `demo/demo.ipynb`.
