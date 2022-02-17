#!/bin/bash
set -e

DATASET_DIR="/gcs/albert-datasets-test/kits19"


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

  # For CPU
  # export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  # export XRT_WORKERS="localservice:0;grpc://localhost:51011"

  # For GPU
  # export GPU_NUM_DEVICES=1

  # For TPU
  export NO_CUDA=1
  export XRT_TPU_CONFIG="localservice;0;localhost:51011"
  # export XRT_TPU_CONFIG="tpu_worker;0;localhost:8470"


  python3 main.py --data_dir $DATASET_DIR \
    --epochs 4000 \
    --evaluate_every 1 \
    --start_eval_at 1 \
    --quality_threshold 0.908 \
    --batch_size 16 \
    --optimizer sgd \
    --ga_steps 1 \
    --learning_rate 0.8 \
    --seed 0 \
    --lr_warmup_epochs 200 \
    --input_shape 64 64 64 \
    --val_input_shape 64 64 64 \
    --torch_xla

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,0,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi