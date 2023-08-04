from super_gradients.training import models
from super_gradients.training.utils.distributed_training_utils import setup_device

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
setup_device(num_gpus=1)
best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/yolonas-hand/ckpt_best.pth").to("cuda")

img_url = 'dataset/GestureData/hypersoft_rgb/'

IMAGES = [
    "dataset/GestureData/hypersoft_rgb/638109209423979966_0_RGB.png",
    "dataset/GestureData/hypersoft_rgb/638109209428273945_0_RGB.png",
]
predictions = best_model.predict(img_url)
# predictions.show()
predictions.save(output_folder="checkpoints/results")