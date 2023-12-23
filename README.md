# Exploring the Impact of IID-ness on Various FL Approaches at Different Stages of Convergence

## where is the report?
The report is in the root directory, named [10719_Project_Report.pdf](https://github.com/ruixCMU/10719-project/blob/main/10_719_Project_Report.pdf).

## Environment Install
Please change the name in the first line of `requirements.yml` (otherwise I believe this will change your default env) and then run:

```
conda env create -f requirements.yml
```

to create an Anaconda environment for you to run my code. I have to apologize for not being familiar with how to describe my environment more briefly. In fact, I think installing:
1. torch==2.0.1 with CUDA (w/o is also OK, but will be slow)
2. numpy ==1.26.1
3. pandas==1.5.3
4. tqdm==4.65.0
5. torchsummary==1.5.1 (this is not necessary but can help you visualize the model)

would be enough.

## Running Experiments
There are four experiments in the report (1-4), corresponding to Figures 2-5, respectively. To run experiment `i`, please:

```
cd bats
experiment[i].bat   # run the bat
cd ../plotters
python plotter[i].py
```

This wil give you plots in the `plots/experiment[i]/` folder.

**Note** I had just re-organized the directory and some coding files, if there are any bugs, please let me know and allow me a bit time to fix it. Sorry for that.

## Pretrain
The pretrained models are in [Google Drive](https://drive.google.com/file/d/1ZpijGdyOhHzn7MyiNt1NfEYDfE_l_O3k/view?usp=sharing). If you want to pretrain them by yourself, you could use the following code:

```
# make sure you are in root directory
python pretrain.py --data_name fmnist --goal [accuracy you want] --max_epochs [maximum epochs] --model MLP --mlp_hidden 512 128
```

This will generate model in `models/` folder.

## Others
You could also tune the hyper-parameters in each file as you wish. Thank you very much!

**Last but not least**, many thanks to **Jong-Ik Park**. His code helped me a lot in working on this project.
