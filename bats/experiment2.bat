@ECHO off
setlocal EnableDelayedExpansion
SET "model_names[0]=MLP_[512, 128]_goal=0.1_max-epochs=100_final-acc=0.0906"
SET "model_names[1]=MLP_[512, 128]_goal=0.5_max-epochs=100_final-acc=0.4923"
SET "model_names[2]=MLP_[512, 128]_goal=0.9_max-epochs=100_final-acc=0.8917"
SET "num_models=2"

SET "aggregations[0]=FedSGD"
SET "aggregations[1]=FedAvg"
SET "aggregations[2]=FedYogi"
SET "aggregations[3]=FedAdaGrad"
SET "aggregations[4]=FedAdam"
SET "num_aggregations=4"

for %%b in (0.001 1 10000) do (
    for /L %%i in (0, 1, %num_models%) do (
        for /L %%j in (0, 1, %num_aggregations%) do (
            if !aggregations[%%j]!==FedSGD (
                SET num_rounds=250
            ) else (
                SET num_rounds=50
            )
            python ../main.py --data_name fmnist --beta %%b --model_name "!model_names[%%i]!" --aggregation !aggregations[%%j]! --num_rounds !num_rounds! --partitioned
        )
    )
)