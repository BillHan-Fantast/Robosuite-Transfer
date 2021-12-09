#!/bin/bash


for r in Panda
do
  for e in Door Lift Door_light Lift_light Door_dark Lift_dark
  do
      export alg=cql
      export env=Door
      export robo=${r}
      export data=medium_expert
      sbatch scripts/iris_offline_a40.slrm
  done
done



