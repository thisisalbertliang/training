#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script, directly run:
#   tpu_run_and_time.sh

SEED=0

MAX_EPOCHS=4000
QUALITY_THRESHOLD="0.908"
SAVE_CKPT_EVERY=100
SAVE_CKPT_DIR_PATH="results"
EVALUATE_EVERY=20
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
DATASET_DIR="/gcs/albert-datasets-test/kits19"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
INPUT_SHAPE=128


if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  python3 -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

    export XRT_TPU_CONFIG="localservice;0;localhost:51011"
    export XLA_USE_BF16=1

  python3 main.py --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --input_shape ${INPUT_SHAPE} ${INPUT_SHAPE} ${INPUT_SHAPE} \
    --save_ckpt_dir_path ${SAVE_CKPT_DIR_PATH} \
    --save_ckpt_every ${SAVE_CKPT_EVERY} \
    --verbose \
    --torch_xla

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi