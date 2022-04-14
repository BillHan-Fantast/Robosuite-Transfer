#!/bin/bash

for e in Sawyer_to_IIWA
do
  for t in Reach Push Lift
  do
    for lr in 0.00005 0.0001 0.0003
    do
      for r in 10 20
      do
        for p in 10.0 50.0 100.0
        do
          for s in 0 1 2
          do
            export alg=vae_cql
            export env=${e}/${t}_light_MR
            export pair=${t}/2k
            export repr=${r}
            export domain=10.0
            export lr=${lr}
            export beta=0.1
            export inverse=10.0
            export paired=${p}
            export seed=${s}
            sbatch scripts/iris_vae_hi.slrm
          done
        done
      done
    done
  done
done

