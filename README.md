# Learning a Better Neural Network Architecture for Multiple Object Tracking

This implementation is based on the **CVPR 2020 (oral)** paper *Learning a Neural Solver for Multiple Object Tracking* ([Guillem Brasó](https://dvl.in.tum.de/team/braso/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/))
[[Paper]](https://arxiv.org/abs/1912.07515)[[Youtube]](https://www.youtube.com/watch?v=YWEirYMaLWc)[[CVPR Daily]](https://www.rsipvision.com/ComputerVisionNews-2020July/55/)
The old implementation addreess is [here](https://github.com/dvl-tum/mot_neural_solver).

Out 2 mechanism is showed below
![Method Visualization](data/pic_1.png)

## Setup

If you want to look the original setup, please read [old_README.md](https://github.com/anslt/ADL4CV/blob/master/old_README.md)

1. Clone and enter this repository:
   ```
   git clone --recursive https://github.com/anslt/ADL4CV.git
   cd mot_neural_solver
   ```
2. (**OPTIONAL**) Download Anaconda if you work on Colab
    ```
    wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    chmod +x Anaconda3-2020.11-Linux-x86_64.sh
    bash ./Anaconda3-2020.11-Linux-x86_64.sh -b -f -p /usr/local
    rm Anaconda3-2020.11-Linux-x86_64.sh
    ```
3. Create an [Anaconda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for this project:
    ```
    conda env create -f environment.yaml
    conda activate mot_neural_solver
    pip install -e tracking_wo_bnw
    pip install -e .
    ```
4. Download the [MOTChallenge data](https://motchallenge.net/), reid network and preprocessed detection:
    ```
    bash scripts/setup/download_motcha.sh
    bash scripts/setup/download_models.sh
    bash scripts/setup/download_prepr_dets.sh
    ```
5. (**OPTIONAL**) Install other lacking package if you work on Colab:
    ```
    conda install -y ipykernel
    ```

## Training
You can train a model by running:
```
python scripts/train.py 
```
By default, sequences `MOT17-04` and `MOT17-11` will be used for validation, and all remaining sequences in the `MOT15`
and `MOT17` datasets will be used for training. You can use other validation sets by
modifying the parameters `data_splits.train` and `data_splits.val`, or use several splits and perform [cross-validation](#Cross-Validation).

In order to train with all available sequences, and reproduce the training of the `MOT17` model we provide, run the following:
```
python scripts/train.py with data_splits.train=all_train train_params.save_every_epoch=True train_params.num_epochs=6
```

For other parameters in training, we introduce below:

graph_model_params: <br />
&nbsp;&nbsp; time_aware: whether the node updating is time aware (defualt: False) <br />
&nbsp;&nbsp; network_split: whether share MLP in different iterations of message pssing (defualt: False) <br />
&nbsp;&nbsp; attention: <br />
&nbsp;&nbsp;&nbsp;&nbsp;    use_attention: whether use attention (defualt: False) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    alpha: LeakyRelu parameter (deafualt: 0.2) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    attention_head_num: the number of attention network applied in MPN (deafualt: 2) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    att_regu: apply regularization on MPN (deafualt: False) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    att_regu_strength: weight of regularization (deafualt: 0.5) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    new_softmax: apply regularization on MPN (deafualt: False)  <br />

&nbsp;&nbsp;  dynamical_graph: <br />
&nbsp;&nbsp;&nbsp;&nbsp;    graph_pruning: whether use graph (deafualt: False)  <br />
&nbsp;&nbsp;&nbsp;&nbsp;    first_prune_step: from which iteration we start prune the graph(deafualt: 4) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    prune_factor: how many edges are pruned in each iteration(deafualt: 0.05) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    prune_frequency: the frequency prune the graph(deafualt: 1) <br />
&nbsp;&nbsp;&nbsp;&nbsp;    mode: the score is generated in which method (deafualt: "classifier node wise") <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;["classifier node wise","classifier naive","similarity node wise","similarity naive"] <br />
&nbsp;&nbsp;&nbsp;&nbsp;    prune_min_edge: the threshold of linked edges to stop pruning for a node (deafualt: 5) <br />

 
## Cross-Validation
As explained in the paper, we perform cross-validation to report the metrics of ablation experiments.
To do so, we divide `MOT17` sequences in 3 sets of train/val splits. For every configuration, we then run
3 trainings, one per validation split, and report the overall metrics.

You can train and evaluate models in this manner by running:
```
RUN_ID=your_config_name
python scripts/train.py with run_id=$RUN_ID cross_val_split=1
python scripts/train.py with run_id=$RUN_ID cross_val_split=2
python scripts/train.py with run_id=$RUN_ID cross_val_split=3
python scripts/cross_validation.py with run_id=$RUN_ID
```
By setting `cross_val_split` to 1, 2 or 3, the training and validation sequences corresponding
to the splits we used in the paper will be set automatically (see `src/mot_neural_solver/data/splits.py`).

The last script will gather the stored metrics from each training run, and compute overall `MOT17 metrics` with them.
This will be done by searching output files containing `$RUN_ID` on them, so it's important that this tag is unique.


## Our Settiings
We use the cross_val_split 2.
We train a model with regularized attention by running:
```
python scripts/train.py with cross_val_split=2 train_params.save_every_epoch=True train_params.num_epochs=25 train_params.num_workers=4 graph_model_params.attention.use_attention=True graph_model_params.attention.att_regu=True graph_model_params.attention.new_softmax=True 
```

We train a model with edge pruning:
```
python scripts/train.py with cross_val_split=2 train_params.save_every_epoch=True train_params.num_epochs=25 train_params.num_workers=4 graph_model_params.dynamical_graph.graph_pruning=True
```





