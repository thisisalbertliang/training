#!/bin/bash
set -e
set -x

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=0

MAX_EPOCHS=100
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=100
EVALUATE_EVERY=500
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
# DATASET_DIR="/home/albertliang/pytorch/data/kits19"
DATASET_DIR="/gcs/albert-datasets-test/kits19"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1


if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

    export XRT_TPU_CONFIG="localservice;0;localhost:51011"
    # export XLA_USE_BF16=1

  # python3 -m torch_xla.distributed.xla_dist \
  #   --tpu=albert5-tpu-v3-32 \
  #   --restart-tpuvm-pod-server \
  #   -- \
  #   python3 /home/albertliang/pytorch/main.py --data_dir ${DATASET_DIR} \
  #   --epochs ${MAX_EPOCHS} \
  #   --evaluate_every ${EVALUATE_EVERY} \
  #   --start_eval_at ${START_EVAL_AT} \
  #   --quality_threshold ${QUALITY_THRESHOLD} \
  #   --batch_size ${BATCH_SIZE} \
  #   --optimizer sgd \
  #   --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
  #   --learning_rate ${LEARNING_RATE} \
  #   --seed ${SEED} \
  #   --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
  #   --input_shape 128 128 128 \
  #   --save_ckpt_dir_path results \
  #   --save_ckpt_every 100 \
  #   --torch_xla

  python3 main.py --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --input_shape 128 128 128 \
    --debug \
    --device xla
    # --profile_port 9111

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