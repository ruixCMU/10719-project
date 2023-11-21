@REM python ../main.py --data_name mnist --beta 0.01 --seed 0 --partitioned
@REM python ../main.py --data_name mnist --beta 0.1 --seed 0 --partitioned
@REM python ../main.py --data_name mnist --beta 1 --seed 0 --partitioned
@REM python ../main.py --data_name mnist --beta 10 --seed 0 --partitioned
@REM python ../main.py --data_name mnist --beta 100 --seed 0 --partitioned
@REM python ../main.py --data_name mnist --beta 1000 --seed 0 --partitioned
@REM python ../main.py --data_name mnist --beta 10000 --seed 0 --partitioned

@REM python ../main.py --data_name cifar10 --beta 0.01 --seed 0 --partitioned
@REM python ../main.py --data_name cifar10 --beta 0.1 --seed 0 --partitioned
@REM python ../main.py --data_name cifar10 --beta 1 --seed 0 --partitioned
python ../main.py --data_name cifar10 --beta 10 --seed 0 --partitioned
python ../main.py --data_name cifar10 --beta 100 --seed 0 --partitioned
python ../main.py --data_name cifar10 --beta 1000 --seed 0 --partitioned
python ../main.py --data_name cifar10 --beta 10000 --seed 0 --partitioned