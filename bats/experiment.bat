@echo off
setlocal enabledelayedexpansion

rem 定义一个数组
set "myArray[0]=Item1"
set "myArray[1]=Item2"
set "myArray[2]=Item3"

rem 获取数组长度
set "arrayLength=3"

rem 遍历数组
for /L %%i in (0, 1, %arrayLength%) do (
    echo Item %%i: !myArray[%%i]!
)

endlocal
