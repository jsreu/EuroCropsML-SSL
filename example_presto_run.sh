#!/bin/bash

PHASE=tuned-finetuning
ALGORITHMNAME=presto  # for run naming
MODEL=presto
DATASOURCE_LIST='["S2"]' # or '["S1","S2"]'; for dataset config
DATASOURCE=$(echo $DATASOURCE_LIST | tr -d '[]" ,' ) # for run naming
BASENAME=EuroCrops_SSL_${ALGORITHMNAME}

for SEED in 42 0 1 123 1234
do
  export EUROCROPS_SSL_SEED=$SEED
  for BATCH_SIZE in 16
  do
    for MAX_SAMPLES in 1 5 10 20 100 200 500
    do
      for LR_SETTING in headbackbonesame headbackbonediff head
      do
        NAME=${DATASOURCE}_${ALGORITHMNAME}_batch${BATCH_SIZE}_max${MAX_SAMPLES}_${LR_SETTING}_seed${SEED}
        eurocropsssl-cli experiments eurocrops ${PHASE} eurocrops_finetuning_maxsamples_${MAX_SAMPLES} base_name=${BASENAME} model=${MODEL} +eurocrops_dataset.data_sources=${DATASOURCE_LIST} finetune=${LR_SETTING} finetune.train_config.batch_size=${BATCH_SIZE} --run-name=${NAME} --pretrain-run=${ALGORITHMNAME}
      done
    done
  done
done
