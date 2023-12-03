@ECHO off
setlocal EnableDelayedExpansion
@REM SET "model_names[0]=MLP_[512, 128]_goal=0.1_max-epochs=100_final-acc=0.0906"
@REM SET "model_names[1]=MLP_[512, 128]_goal=0.2_max-epochs=100_final-acc=0.2227"
@REM SET "model_names[2]=MLP_[512, 128]_goal=0.3_max-epochs=100_final-acc=0.2916"
@REM SET "model_names[3]=MLP_[512, 128]_goal=0.4_max-epochs=100_final-acc=0.3827"
@REM SET "model_names[4]=MLP_[512, 128]_goal=0.5_max-epochs=100_final-acc=0.4923"
@REM SET "model_names[5]=MLP_[512, 128]_goal=0.6_max-epochs=100_final-acc=0.5908"
@REM SET "model_names[6]=MLP_[512, 128]_goal=0.7_max-epochs=100_final-acc=0.6873"
@REM SET "model_names[7]=MLP_[512, 128]_goal=0.8_max-epochs=100_final-acc=0.797"
@REM SET "model_names[8]=MLP_[512, 128]_goal=0.9_max-epochs=100_final-acc=0.8917"
@REM SET "num_models=8"

SET "model_names[0]=MLP_[512, 128]_goal=0.1_max-epochs=100_final-acc=0.0906"
SET "model_names[1]=MLP_[512, 128]_goal=0.5_max-epochs=100_final-acc=0.4923"
SET "model_names[2]=MLP_[512, 128]_goal=0.9_max-epochs=100_final-acc=0.8917"
SET "num_models=2"

@REM SET "aggregations[0]=FedAdam"
@REM SET "aggregations[1]=FedAdaGrad"
SET "aggregations[0]=FedYogi"
SET "num_aggregations=0"

@ECHO on
for %%b in (0.001 1 10000) do (
    for /L %%i in (0, 1, %num_models%) do (
        for /L %%j in (0, 1, %num_aggregations%) do (
            python ../main.py --data_name fmnist --beta %%b --model_name "!model_names[%%i]!" --aggregation !aggregations[%%j]! --partitioned
        )
    )
)