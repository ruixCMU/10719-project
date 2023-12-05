@ECHO off
setlocal EnableDelayedExpansion
SET "model_names[0]=MLP_[512, 128]_goal=0.3_max-epochs=100_final-acc=0.2916"
SET "model_names[1]=MLP_[512, 128]_goal=0.6_max-epochs=100_final-acc=0.5908"
SET "model_names[2]=MLP_[512, 128]_goal=0.9_max-epochs=100_final-acc=0.8917"
SET "num_models=2"

for %%b in (0.01 10000) do (
    for /L %%i in (0, 1, %num_models%) do (
        python ../main.py --data_name fmnist --beta %%b --model_name "!model_names[%%i]!" --case 1
        python ../main.py --data_name fmnist --beta %%b --model_name "!model_names[%%i]!" --case 2
    ) 
)

endlocal
