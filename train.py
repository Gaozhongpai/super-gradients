
##
from super_gradients.training import Trainer

from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import models

from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

from super_gradients.common import MultiGPUMode
from super_gradients.training.utils.distributed_training_utils import setup_device

from super_gradients.training.utils.detection_utils import DetectionVisualization

import torch 
import numpy as np

# from roboflow import Roboflow
# rf = Roboflow(api_key="BlKteKodU1zl1tXme02r")
# project = rf.workspace("atathamuscoinsdataset").project("dataset/u.s.-coins-dataset-a.tatham")
# dataset = project.version(5).download("yolov5")


def my_undo_image_preprocessing(im_tensor: torch.Tensor):
    im_np = im_tensor.cpu().numpy()
    im_np = im_np[:, ::-1, :, :].transpose(0, 2, 3, 1)
    im_np *= 255.0

    return np.ascontiguousarray(im_np, dtype=np.uint8)


setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=2)

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

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':8
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':8
    }
)

# test_data = coco_detection_yolo_format_val(
#     dataset_params={
#         'data_dir': dataset_params['data_dir'],
#         'images_dir': dataset_params['test_images_dir'],
#         'labels_dir': dataset_params['test_labels_dir'],
#         'classes': dataset_params['classes']
#     },
#     dataloader_params={
#         'batch_size':8,
#         'num_workers':2
#     }
# )




train_data.dataset.transforms
train_data.dataset.plot()
imgs, targets = next(iter(train_data))
# DetectionVisualization.visualize_batch(imgs, preds, targets, batch_name='train', class_names=['p', 'h'],
#                                        checkpoint_dir='checkpoints/yolonas-hand/', gt_alpha=0.5,
#                                        undo_preprocessing_func=my_undo_image_preprocessing)

CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='yolonas-hand', ckpt_root_dir=CHECKPOINT_DIR)

model = models.get('yolo_nas_l',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
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

trainer.train(model=model,
              training_params=train_params,
              train_loader=train_data,
              valid_loader=val_data)

# ## inference
# best_model = models.get('yolo_nas_l',
#                         num_classes=len(dataset_params['classes']),
#                         checkpoint_path="checkpoints/my_first_yolonas_run/average_model.pth")
# # test
# trainer.test(model=best_model,
#             test_loader=test_data,
#             test_metrics_list=DetectionMetrics_050(score_thres=0.1,
#                                                    top_k_predictions=300,
#                                                    num_cls=len(dataset_params['classes']),
#                                                    normalize_targets=True,
#                                                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
#                                                                                                           nms_top_k=1000,
#                                                                                                           max_predictions=300,
#                                                                                                           nms_threshold=0.7)))
# # predict                                            
# img_url = 'https://www.mynumi.net/media/catalog/product/cache/2/image/9df78eab33525d08d6e5fb8d27136e95/s/e/serietta_usa_2_1/www.mynumi.net-USASE5AD160-31.jpg'
# best_model.predict(img_url).show()