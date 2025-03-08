sudo apt-get install python3-venv
python3 -m venv env
source env/bin/activate
pip3 install -e '.[jax_cpu]'
pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/cu121'
pip3 install -e '.[full]'
python3 datasets/dataset_setup.py --data_dir=~/data --ogbg
python3 datasets/dataset_setup.py --data_dir=~/data --wmt
python3 datasets/dataset_setup.py --data_dir=~/data --fastmri --fastmri_knee_singlecoil_train_url '<knee_singlecoil_train_url>' --fastmri_knee_singlecoil_val_url '<knee_singlecoil_val_url>' --fastmri_knee_singlecoil_test_url '<knee_singlecoil_test_url>'
ulimit -n 8192
python3 datasets/dataset_setup.py --data_dir=~/data --imagenet --temp_dir=~/data/tmp --imagenet_train_url '<imagenet_train_url>' --imagenet_val_url '<imagenet_val_url>' --framework pytorch
python3 datasets/dataset_setup.py --data_dir=~/data --temp_dir=~/data/tmp --criteo1tb
python3 datasets/dataset_setup.py --data_dir=~/data --temp_dir=~/data/tmp --librispeech

pip install "flax[all]"
pip install clu

pip install wandb tensorflow-datasets
pip install 'setuptools<64.0.0'
pip install promise
pip install tensorflow-datasets

torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=N_GPUS \
    submission_runner.py \
    --framework=pytorch \
    --workload=<workload> \
    --experiment_dir=<path_to_experiment_dir> \
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>


python3 submission_runner.py \
    --framework=pytorch \
    --workload=mnist \
    --experiment_dir=/home/ubuntu/algoperf_experiments \
    --experiment_name=my_first_experiment \
    --submission_path=submissions/my_submissions/submission.py \
    --tuning_search_space=submissions/my_submissions/tuning_search_space.json