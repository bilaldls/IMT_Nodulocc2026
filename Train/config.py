# config.py
config = {
    "batch_size": 8,
    "num_workers": 2,
    "image_size": 224,
    "data_fraction": 0.3,  # 1.0 = 100% des données, 0.5 = 50%, etc.
    "cache_dir": "feature_cache",
    "phase1_epochs": 10,
    "phase2_epochs": 3,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "output_dir": "checkpoints_multitask_fast",
}
