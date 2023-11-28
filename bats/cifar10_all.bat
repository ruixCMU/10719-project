@ECHO off
setlocal EnableDelayedExpansion
SET "model_names[0]=ResNet18_goal=0.5_max-epochs=100_final-acc=0.4865"
SET "model_names[1]=ResNet18_goal=0.6_max-epochs=100_final-acc=0.5839"
SET "model_names[2]=ResNet18_goal=0.7_max-epochs=100_final-acc=0.6914"
SET "model_names[3]=ResNet18_goal=0.8_max-epochs=100_final-acc=0.802"
SET "model_names[4]=ResNet18_goal=0.9_max-epochs=100_final-acc=0.8822"
SET "num_models=4"

@ECHO on
for %%b in (0.001 0.01 0.1 1 10 100 1000 10000) do (
    for /L %%i in (0, 1, %num_models%) do (
        python ../main.py --data_name cifar10 --beta %%b --model_name !model_names[%%i]!
    )
)