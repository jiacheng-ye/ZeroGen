home_dir="/nvme/yjc/ZeroGen"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset=gpt2-xl_topk0_topp0.9_sst-2-x2
task_name=sst-2
task_type=TC

for task_name in sst-2
do
  cmd="python scripts/self_bleu.py --logto ${task_name}_bleu.log"
  if [ "${task_type}" = "QA" ]; then
    cmd+=" --gen_column question --file out-${task_name}-x2/${dataset}"
  else
    cmd+=" --gen_column X --file out-${task_name}-x2/${dataset}/${task_name}-dataset.jsonl"
  fi

  echo ${cmd}
  eval ${cmd}
done

# standard dataset, we evaluate question for QA, text for TC
cmd="python scripts/self_bleu.py --file data/${task_name}/train --logto ${task_name}_bleu.log"
if [ "${task_type}" = "QA" ]; then
  cmd+=" --gen_column question"
else
  cmd+=" --gen_column text"
fi

echo ${cmd}
eval ${cmd}
