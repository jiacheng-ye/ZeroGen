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
train_batch_size=32  # for train the small model

task=squad
#task=adversarial_qa

top_k=0
top_p=0.9
num_entries_per_input=2
log_every=10000 # train the small model after generating #log_every examples

#for model_name in gpt2-xl gpt2-large gpt2
for task in squad adversarial_qa
do
  echo "############################################# Supervised with Human Annotations ###################################################"
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/misc.py \
  --task_name ${task} \
  --train_batch_size ${train_batch_size} \
  --small_model_name ${small_model_name}
  "
  echo ${cmd}
  eval ${cmd}

  echo "############################################# Prompting with PLM (Zero-shot performance) ###################################################"
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
  --input_file data/${task} \
  --output_dir out-${task} \
  --task_file tasks/${task}/${task}-zero-shot.json \
  --batch_size ${batch_size} \
  --model_name ${model_name}
  "
  echo ${cmd}
  eval ${cmd}

  echo "############################################# Generating Answer Y ###################################################"
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
   --output_dir out-${task}-x1 \
   --task_file tasks/${task}/${task}-x1.json \
   --model_name ${model_name}
   "

  echo ${cmd}
  eval ${cmd}

  echo "############################################# Generating Question X with PLM ###################################################"
  cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
   --model_name ${model_name}
   --output_dir out-${task}-x2 \
   --task_file tasks/${task}/${task}-x2.json \
   --input_file out-${task}-x1 \
   --num_entries_per_input ${num_entries_per_input} \
   --batch_size ${batch_size} \
   --top_k ${top_k} \
   --top_p ${top_p} \
   --small_model_name ${small_model_name} \
   --min_length 5 \
   --max_length 40 \
   --train_batch_size ${train_batch_size} \
   --log_every ${log_every}
   "
  echo ${cmd}
  eval ${cmd}

done
