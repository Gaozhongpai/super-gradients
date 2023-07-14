from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from super_gradients.module_interfaces import HasPredict


def fuse_conv_bn(model: nn.Module, replace_bn_with_identity: bool = False):
    """
    Fuses consecutive nn.Conv2d and nn.BatchNorm2d layers recursively inplace in all of the model
    :param replace_bn_with_identity: if set to true, bn will be replaced with identity. otherwise, bn will be removed
    :param model: the target model
    :return: the number of fuses executed
    """
    children = list(model.named_children())
    counter = 0
    for i in range(len(children) - 1):
        if isinstance(children[i][1], torch.nn.Conv2d) and isinstance(children[i + 1][1], torch.nn.BatchNorm2d):
            setattr(model, children[i][0], torch.nn.utils.fuse_conv_bn_eval(children[i][1], children[i + 1][1]))
            if replace_bn_with_identity:
                setattr(model, children[i + 1][0], nn.Identity())
            else:
                delattr(model, children[i + 1][0])
            counter += 1
    for child_name, child in children:
        counter += fuse_conv_bn(child, replace_bn_with_identity)

    return counter


def infer_format_from_file_name(output_filename: str) -> Optional[str]:
    if isinstance(output_filename, str) and output_filename.endswith(".onnx"):
        return "onnx"

    return None


def infer_image_shape_from_model(model: Union[nn.Module, HasPredict]) -> Optional[Tuple[int, int]]:
    """
    Infer the image shape from the model. This function takes the preprocessing parameters if they are available
    and gets the input image shape from them. If the preprocessing parameters are not available, the function returns None
    :param model:
    :return: A tuple of (height, width) or None
    """
    if not isinstance(model, HasPredict):
        return None

    return None
