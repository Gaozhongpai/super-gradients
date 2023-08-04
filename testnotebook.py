import super_gradients
from torchinfo import summary

yolo_nas_l = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()
# yolo_nas_l.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg").show()

summary(model=yolo_nas_l,
        input_size=(16, 3, 640, 640),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
