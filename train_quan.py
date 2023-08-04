from super_gradients.common import MultiGPUMode
from super_gradients.training.utils.distributed_training_utils import setup_device

from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, \
                                                        coco_detection_yolo_format_train_params, \
                                                        coco_detection_yolo_format_val, \
                                                        coco_detection_yolo_format_val_params
import os
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.pre_launch_callbacks import modify_params_for_qat

# setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=2)

dataset_params = {
    'data_dir':'dataset/coco',
    'train_images_dir':'images/train2017',
    'train_labels_dir':'labels/coco_wholebody_train2017',
    'val_images_dir':'images/val2017',
    'val_labels_dir':'labels/coco_wholebody_val2017',
    'test_images_dir':'images/val2017',
    'test_labels_dir':'labels/coco_wholebody_val2017',
    'classes': ['p', 'h']
}

model = models.get('yolo_nas_l',
                    num_classes=len(dataset_params['classes']),
                    checkpoint_path="checkpoints/yolonas-hand/ckpt_best.pth").to("cuda")


train_dataset_params, train_dataloader_params = coco_detection_yolo_format_train_params(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':64,
        'num_workers':16
    }
)


val_dataset_params, val_dataloader_params = coco_detection_yolo_format_val_params(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':64,
        'num_workers':16
    }
)


train_params = {
    # ENABLING SILENT MODE
    'silent_mode': True,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 100,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
    train_params, 
    train_dataset_params, 
    val_dataset_params,
    train_dataloader_params, 
    val_dataloader_params
)

train_data = coco_detection_yolo_format_train(
    dataset_params=train_dataset_params,
    dataloader_params=train_dataloader_params
)

val_data = coco_detection_yolo_format_val(
    dataset_params=val_dataset_params,
    dataloader_params=val_dataloader_params
)


CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name="yolonas-hand-quan", ckpt_root_dir=CHECKPOINT_DIR)

trainer.qat(model=model, 
            training_params=train_params, 
            train_loader=train_data, 
            valid_loader=val_data, 
            calib_loader=train_data)