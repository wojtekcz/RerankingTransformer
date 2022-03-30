python experiment_rerank.py -F logs/train_rerank_finetune_r50_s2s with \
      temp_dir=logs/train_rerank_finetune_r50_s2s \
      dataset.s2s_rerank model.resnet50 model.freeze_backbone=False \
      cache_nn_inds=rrt_sop_caches/rrt_r50_s2s_nn_inds_test.pkl
