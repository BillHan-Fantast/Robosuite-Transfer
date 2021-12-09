#!/bin/bash

for e in Lift Door
do
    export env=$e
    export robot=Panda
    export ctrl=OSC_POSE
    export var=frontview
    export plr=0.001
    sbatch scripts/iris_ft.slrm
done