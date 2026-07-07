#!/bin/sh

#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:15:00
#PJM -L jobenv=singularity
#PJM -j

module load singularity-ce

singularity build fluid_analysis_util.sif Singularity.def
singularity run -B /fast/jh240062:/data fluid_analysis_util.sif