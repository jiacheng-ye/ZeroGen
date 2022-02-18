home_dir="/nvme/yjc/ZeroGen"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export WANDB_DISABLED=true # disable wandb in huggingface transformers
#export TRANSFORMERS_OFFLINE=1 # uncomment this line if you have downloaded the transformer models, it tells Transformers to use local files only and will not try to look things up.
export WANDB_PROJECT=ZeroGen  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key


gpu=1
model_name=gpt2-xl
small_model_name=distilbert-base-uncased

batch_size=32 # for generation with PLM
train_batch_size=32  # for train the small model (DistilBERT by default)

#task=sst-2

################################################################
for task in rte
do
  echo "############################################# Supervised with Human Annotations ###################################################"
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/misc.py \
  --task_name ${task} \
  --small_model_name ${small_model_name} \
  --train_batch_size ${train_batch_size}
  "
  echo ${cmd}
  eval ${cmd}


  echo "############################################# Prompting with PLM (Zero-shot performance) ###################################################"
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
  --output_dir out-${task} \
  --task_file tasks/${task}/${task}-zero-shot.json \
  --batch_size ${batch_size} \
  --model_name ${model_name}
  "
  echo ${cmd}
  eval ${cmd}


  echo "############################################# Generating Context C with PLM ###################################################"
  top_k=0
  top_p=0.9
  num_entries_per_input=800000

  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
   --output_dir out-${task}-x1 \
   --task_file tasks/${task}/${task}-x1.json \
   --num_entries_per_input ${num_entries_per_input} \
   --top_k ${top_k} \
   --top_p ${top_p} \
   --batch_size 512 \
   --max_length 10"

  echo ${cmd}
  # comment next line for NLI tasks, as we use the given rather than generated context/premise
  eval ${cmd}


  echo "############################################# Generating X with PLM ###################################################"
  top_k=0
  top_p=0.9
  num_entries_per_input=32
  log_every=10000 # train the small model after generating #log_every examples

  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
   --model_name ${model_name}
   --output_dir out-${task}-x2 \
   --task_file tasks/${task}/${task}-x2.json \
   --num_entries_per_input ${num_entries_per_input} \
   --batch_size ${batch_size} \
   --train_batch_size ${train_batch_size} \
   --top_k ${top_k} \
   --top_p ${top_p} \
   --small_model_name ${small_model_name} \
   --min_length 10 \
   --max_length 40 \
   --log_every ${log_every}
   "
   # using generated x1 for sst-2 and imdb, while using gold x1 for rte and qnli
  if [ "${task}" = "sst-2" ] || [ "${task}" = "imdb" ]; then
    cmd+=" --input_file out-${task}-x1/${task}-dataset.jsonl"
  else # using self-debiasing for nli
    cmd+=" --decay_constant 200"
  fi

  echo ${cmd}
  eval ${cmd}

done