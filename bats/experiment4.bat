@ECHO off
setlocal EnableDelayedExpansion

SET "model_name=MLP_[512, 128]_goal=0.5_max-epochs=100_final-acc=0.4923"

for %%r in (0.2 0.4 0.6 0.8) do (
    python ../main.py --data_name fmnist --beta 0.001 --model_name "!model_name!" --aggregation FedAvg --ratio_mixed_iid %%r
    python ../main.py --data_name fmnist --beta 0.001 --model_name "!model_name!" --aggregation FedAdam --ratio_mixed_iid %%r
)

endlocal
