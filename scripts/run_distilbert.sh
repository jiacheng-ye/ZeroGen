home_dir="/nvme/yjc/ZeroGen"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export WANDB_DISABLED=true
#export TRANSFORMERS_OFFLINE=1 # uncomment this line if you have downloaded the transformer models, it tells Transformers to use local files only and will not try to look things up.

task_name=sst-2  # sst-2, imdb, qnli, rte, squad, adversarial_qa
task_type=TC  # TC, NLI, QA
dataset=gpt2-xl_topk0_topp0.9_sst-2-x2
#limit=100000
gpu=3

small_model_name=distilbert-base-uncased
small_model_ckpt=
no_train=false  # whether to train the small model or directly use the checkpoint to validate

for limit in 10000
do
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python scripts/misc.py \
  --task_name ${task_name} \
  --small_model_name ${small_model_name} \
  --output_dir tmp"

  if [ "${limit}" != "" ] ; then
    cmd+=" --limit ${limit}"
  fi

  input_dir=${home_dir}/out-${task_name}-x2/${dataset}
  if [ "${dataset}" != "" ] ; then
      if [ "${task_type}" = "QA" ]; then
        cmd+=" --dataset ${input_dir}"
      else
        cmd+=" --dataset ${input_dir}/${task_name}-dataset.jsonl"
      fi
  fi

  if [ "${no_train}" = true ] ; then
    cmd+=" --no_train"
  fi

  if [ "${small_model_ckpt}" != "" ] ; then
    cmd+=" --small_model_ckpt ${small_model_ckpt}"
  fi

  echo ${cmd}
  eval ${cmd}
done

