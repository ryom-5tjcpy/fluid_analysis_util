#!bin/sh

#PJM -L rscgrop=a-batch
#PJM -L node=1
#PJM -L elapse=0:15:00
#PJM -L jobenv=singularity
#PJM -j

if [-z $SINGULARITY_DOCKER_USERNAME]; then
    echo "UNDEFINED SINGULARITY_DOCKER_USERNAME"
    exit 1
fi

if [-z $SINGULARITY_DOCKER_PASSWORD]; then
    echo "UNDEFINED SINGULARITY_DOCKERPASSWORD"
    exit 2
fi

singularity build fluid_analysis_util.sif docker://ghcr.io./$SINGULARITY_DOCKER_USERNAME/fluid_analysis_util:latest
singularity run fluid_analysis_util.sif