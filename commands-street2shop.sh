# try to consume street2shop dresses dataset
# ==========================================

conda activate rrt2

cd RRT_SOP

# train global 100 epochs
python experiment_global.py -F logs/train_global_r50_s2s with \
      temp_dir=logs/train_global_r50_s2s \
      dataset.sop_global_s2s model.resnet50

# INFO - Global (train) - Started run with ID "8"

# initial
# {1: 77.72, 10: 88.93, 100: 93.58, 1000: 97.78}
# Validation [000]
# {1: 84.61, 10: 96.2, 100: 98.69, 1000: 99.62}
# Validation [001]
# {1: 85.52, 10: 96.41, 100: 98.55, 1000: 99.66}
# Validation [002]
# {1: 85.98, 10: 96.59, 100: 98.71, 1000: 99.65}
# Validation [003]
# {1: 85.75, 10: 96.49, 100: 98.69, 1000: 99.7}
# Validation [004]
# {1: 86.25, 10: 96.86, 100: 98.79, 1000: 99.75}
# Validation [005]
# {1: 85.88, 10: 96.99, 100: 98.84, 1000: 99.72}
# Validation [006]
# {1: 86.5, 10: 96.97, 100: 98.92, 1000: 99.69}
# Validation [007]
# {1: 86.12, 10: 96.92, 100: 98.8, 1000: 99.71}
# Validation [008]
# {1: 85.98, 10: 97.02, 100: 98.97, 1000: 99.77}
# Validation [011]
# {1: 86.38, 10: 96.83, 100: 98.92, 1000: 99.72}
# Validation [014]
# {1: 86.29, 10: 96.81, 100: 98.73, 1000: 99.66}

# vs. s2s dresses training
# val
# {1: 84.27, 10: 91.74}
# test
# {1: 72.54, 10: 82.87}

# + configure recalls @ points: 1, 3, 5, 10, 20
# - configure 384
# + make sure we pick first image only as anchor for test
# + is training generating tensorboard logs? no

# naive test query/gallery split
# INFO - Global (train) - Started run with ID "10"

# initial
# {1: 16.26, 3: 29.04, 5: 39.54, 10: 58.42, 20: 74.65}
# Validation [000]
# {1: 31.5, 3: 46.38, 5: 56.9, 10: 72.1, 20: 88.68}
# Validation [001]
# {1: 32.48, 3: 46.84, 5: 56.19, 10: 72.5, 20: 87.81}
# Validation [002]
# {1: 32.68, 3: 47.49, 5: 56.99, 10: 73.52, 20: 89.86}
# Validation [003]
# {1: 33.35, 3: 49.95, 5: 60.02, 10: 76.34, 20: 90.62}

# better test query/gallery split
# INFO - Global (train) - Started run with ID "13"
# initial
# {1: 9.7, 3: 23.48, 5: 34.81, 10: 55.16, 20: 72.66}
# Validation [000]
# {1: 27.39, 3: 44.83, 5: 56.68, 10: 74.79, 20: 90.16}
# Validation [066]
# {1: 29.23, 3: 45.41, 5: 55.79, 10: 73.03, 20: 88.76}

# train set/query set split
# INFO - Global (train) - Started run with ID "14"
# initial
# {1: 9.7, 3: 23.48, 5: 34.81, 10: 55.16, 20: 72.66}
# Validation [000]
# {1: 25.78, 3: 41.17, 5: 51.71, 10: 70.58, 20: 87.8}
# Validation [001]
# {1: 26.35, 3: 42.23, 5: 52.72, 10: 70.37, 20: 87.33}
# Validation [002]
# {1: 26.54, 3: 43.25, 5: 55.03, 10: 73.36, 20: 88.8}
# Validation [004]
# {1: 27.01, 3: 42.71, 5: 54.53, 10: 73.43, 20: 90.66}
# Validation [012]
# {1: 26.4, 3: 43.03, 5: 53.82, 10: 70.86, 20: 87.03}
# Validation [037]
# {1: 26.95, 3: 43.77, 5: 54.23, 10: 71.45, 20: 88.2}

# switched first with remaining samples for eval, disabled samples splitting for train
# INFO - Global (train) - Started run with ID "15"

# initial
# {1: 11.2, 3: 19.11, 5: 22.9, 10: 30.81, 20: 36.08}
# Validation [000]
# {1: 56.34, 3: 67.05, 5: 72.32, 10: 77.76, 20: 82.37}
# Validation [001]
# {1: 57.66, 3: 67.55, 5: 71.33, 10: 77.76, 20: 82.04}
# Validation [002]
# {1: 60.79, 3: 68.86, 5: 72.98, 10: 77.43, 20: 82.54}
# Validation [004]
# {1: 58.48, 3: 70.02, 5: 74.96, 10: 79.9, 20: 82.87}
# Validation [007]
# {1: 61.45, 3: 72.32, 5: 76.28, 10: 80.4, 20: 84.18}

# set crops and resizes to 384
# decreased batch_size = 76, test_batch_size = 76
# INFO - Global (train) - Started run with ID "20"

# initial
# {1: 11.86, 3: 16.47, 5: 21.75, 10: 26.03, 20: 31.47
# Validation [000]
# {1: 46.29, 3: 55.85, 5: 59.97, 10: 66.56, 20: 71.83}
# Validation [001]
# {1: 51.24, 3: 62.93, 5: 68.37, 10: 74.14, 20: 77.76}
# Validation [002]
# {1: 51.24, 3: 60.63, 5: 65.4, 10: 70.51, 20: 77.92}
# Validation [003]
# {1: 54.53, 3: 65.07, 5: 69.69, 10: 74.14, 20: 78.75}

# use valid.txt, increased test_batch_size = 256, fixed num_identities, num_iterations
# INFO - Global (train) - Started run with ID "22"
# initial
# {1: 11.86, 3: 16.47, 5: 21.58, 10: 26.03, 20: 31.47}
# Validation [000]
# {1: 51.24, 3: 61.29, 5: 65.07, 10: 71.33, 20: 75.78}
# Validation [001]
# {1: 52.06, 3: 60.96, 5: 64.42, 10: 70.84, 20: 76.11}
# Validation [002]
# {1: 53.87, 3: 65.57, 5: 69.69, 10: 73.97, 20: 80.4}
# Validation [003]
# {1: 57.17, 3: 66.72, 5: 71.17, 10: 75.95, 20: 81.22}
# Validation [004]
# {1: 57.99, 3: 67.55, 5: 71.66, 10: 77.92, 20: 83.2}
# Validation [006]
# {1: 55.35, 3: 67.55, 5: 72.49, 10: 77.59, 20: 81.88}
# Validation [018]
# {1: 58.81, 3: 67.38, 5: 72.49, 10: 76.61, 20: 82.21}
# Validation [022]
# {1: 60.63, 3: 69.03, 5: 73.15, 10: 78.91, 20: 83.86}
# Validation [023]
# {1: 57.33, 3: 68.04, 5: 71.5, 10: 75.62, 20: 80.72}
# ...
# Validation [036]
# {1: 57.33, 3: 66.72, 5: 70.84, 10: 75.62, 20: 81.38}
# Validation [049]
# {1: 55.68, 3: 66.39, 5: 71.66, 10: 78.09, 20: 82.87}


# prepare indices
# create test ids
python eval_global.py -F logs/eval_global_r50_s2s with temp_dir=logs/eval_global_r50_s2s \
      resume=rrt_sop_ckpts/resnet50_s2s_global.pt dataset.s2s_global model.resnet50

cp logs/eval_global_r50_s2s/nn_inds.pkl rrt_sop_caches/rrt_r50_s2s_nn_inds_test.pkl

# create train ids
python eval_global.py -F logs/nn_file_for_training_s2s with temp_dir=logs/nn_file_for_training_s2s \
      resume=rrt_sop_ckpts/resnet50_s2s_global.pt dataset.s2s_global model.resnet50 \
      query_set='train'

cp logs/nn_file_for_training_s2s/nn_inds.pkl rrt_sop_caches/rrt_r50_s2s_nn_inds_train.pkl


# train reranking (finetuned backbone)
# train-reranking-s2s.sh
python experiment_rerank.py -F logs/train_rerank_finetune_r50_s2s with \
      temp_dir=logs/train_rerank_finetune_r50_s2s \
      dataset.s2s_rerank model.resnet50 model.freeze_backbone=False 

# configure 'user.name' and 'user.email' in git.
