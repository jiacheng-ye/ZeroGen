home_dir="/nvme/yjc/ZeroGen"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

task_name=sst-2  # sst-2, imdb, qnli, rte, squad, adversarial_qa
task_type=TC  # TC, NLI, QA
dataset=gpt2-xl_topk0_topp0.9_sst-2-x2
#limit=100000
gpu=3

for limit in 1000
do
  input_dir=${home_dir}/out-${task_name}-x2/${dataset}
  output_dir=${home_dir}/lstms/${task_type}/data/${task_name}/${dataset}

  # adapt the format of generated dataset to that of LSTM models used here (train and val)
  cmd="python scripts/convert_to_lstm_dataset.py ${task_name} ${input_dir} ${output_dir} ${limit}"
  echo ${cmd}
  eval ${cmd}

  # run LSTM models
  cd lstms/${task_type}
  if [ "${task_type}" = "QA" ]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} python run.py
    --train-file ${output_dir}/train-${limit}.json
    --dev-file ${output_dir}/dev-${limit}.json
    --test-file data/${task_name}/std/test.json
    "
  else
    # copy the test file to output_dir (test set)
    cp data/${task_name}/std/test.jsonl ${output_dir}/
    cmd="CUDA_VISIBLE_DEVICES=${gpu} python run.py ${output_dir} ${task_name} ${limit}"
  fi
  echo ${cmd}
  eval ${cmd}
  cd ../../

done

