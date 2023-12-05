# 10719-project

## where is the report?
The report is in the root directory, named `10719_Project_Report.pdf`

## Running Experiments
There are four experiments in the report (1-4), corresponding to Figures 2-5, respectively. To run experiment `i`, please:

```
cd bats
experiment[i].bat   # run the bat
cd ..
python plotter[i].py
```

This wil give you plots in the `plots/experiment[i]/` folder.

**Note** I had just re-organized the directory and some coding files, if there are any bugs, please let me know and allow me a bit time to fix it. Sorry for that.

The pretrained models are in [Google Drive](https://drive.google.com/file/d/1ZpijGdyOhHzn7MyiNt1NfEYDfE_l_O3k/view?usp=sharing). If you want to pretrain them by yourself, you could use the following code:

```
# make sure you are in root directory
python pretrain.py --data_name fmnist --goal [accuracy you want] --max_epochs [maximum epochs] --model MLP --mlp_hidden 512 128
```

This will generate model in `models/` folder.

What is more, you could tune the hyper-parameters in each file as you wish. Thank you very much!