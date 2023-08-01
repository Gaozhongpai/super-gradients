import abc
import copy
import dataclasses
import gc
from typing import Any
from typing import Union, Optional, List, Tuple

import onnxsim
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.conversion_enums import ExportTargetBackend, ExportQuantizationMode, DetectionOutputFormatMode
from super_gradients.conversion.onnx.nms import attach_onnx_nms
from super_gradients.conversion.preprocessing_modules import CastTensorTo
from super_gradients.conversion.tensorrt.nms import attach_tensorrt_nms
from super_gradients.training.utils.export_utils import infer_format_from_file_name, infer_image_shape_from_model, infer_image_input_channels
from super_gradients.training.utils.quantization.fix_pytorch_quantization_modules import patch_pytorch_quantization_modules_if_needed
from super_gradients.training.utils.utils import infer_model_device
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

logger = get_logger(__name__)

__all__ = ["ExportableObjectDetectionModel", "AbstractObjectDetectionDecodingModule", "ModelExportResult"]


class AbstractObjectDetectionDecodingModule(nn.Module):
    """
    Abstract class for decoding outputs from object detection models to a tuple of two tensors (boxes, scores)
    """

    @abc.abstractmethod
    def forward(self, predictions: Any) -> Tuple[Tensor, Tensor]:
        """
        The implementation of this method must take raw predictions from the model and convert / postprocess them
        to output candidates for NMS. This method may filter out predictions based on confidence threshold and
        it must obey the contract that the number of predictions per image is fixed and equal to
        value returned by self.get_num_pre_nms_predictions().

        :param predictions: Input predictions from the model itself.
        The value of this argument is model-specific

        :return: Implementation of this method must return a tuple of two tensors (boxes, scores) with
        the following semantics:
        - boxes - [B, N, 4]
        - scores - [B, N, C]
        Where N is the maximum number of predictions per image (see self.get_num_pre_nms_predictions()),
        and C is the number of classes.

        """
        raise NotImplementedError(f"forward() method is not implemented for class {self.__class__.__name__}. ")

    def get_output_names(self) -> List[str]:
        """
        Returns the names of the outputs of the module.
        Usually you don't need to override this method.
        Export API uses this method internally to give meaningful names to the outputs of the exported model.

        :return: A list of output names.
        """
        return ["pre_nms_bboxes_xyxy", "pre_nms_scores"]

    @abc.abstractmethod
    def get_num_pre_nms_predictions(self) -> int:
        """
        Returns the number of predictions per image that this module produces.
        :return: Number of predictions per image.
        """
        raise NotImplementedError(f"get_num_pre_nms_predictions() method is not implemented for class {self.__class__.__name__}. ")


@dataclasses.dataclass
class ModelExportResult:
    """
    A dataclass that holds the result of model export.
    """

    input_image_channels: int
    input_image_dtype: torch.dtype
    input_image_shape: Tuple[int, int]

    engine: ExportTargetBackend
    quantization_mode: Optional[ExportQuantizationMode]

    output: str
    output_predictions_format: DetectionOutputFormatMode


class ExportableObjectDetectionModel:
    """
    A mixin class that adds export functionality to the object detection models.
    Classes that inherit from this mixin must implement the following methods:
    - get_decoding_module()
    - get_preprocessing_callback()
    Providing these methods are implemented correctly, the model can be exported to ONNX or TensorRT formats
    using model.export(...) method.
    """

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractObjectDetectionDecodingModule:
        """
        Gets the decoding module for the object detection model.
        This method must be implemented by the derived class and should return
        an instance of AbstractObjectDetectionDecodingModule that would take raw models' outputs and
        convert them to a tuple of two tensors (boxes, scores):
         - boxes: [B, N, 4] - All predicted boxes in (x1, y1, x2, y2) format.
         - scores: [B, N, C] - All predicted scores ([0..1] range) for each box and class.
        :return: An instance of AbstractObjectDetectionDecodingModule
        """
        raise NotImplementedError(f"get_decoding_module() is not implemented for class {self.__class__.__name__}.")

    def get_preprocessing_callback(self, **kwargs) -> Optional[nn.Module]:
        raise NotImplementedError(f"get_preprocessing_callback is not implemented for class {self.__class__.__name__}.")

    def export(
        self,
        output: str,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        engine: Optional[ExportTargetBackend] = None,
        quantization_mode: ExportQuantizationMode = Optional[None],
        selective_quantizer: Optional[SelectiveQuantizer] = None,
        calibration_loader: Optional[DataLoader] = None,
        preprocessing: Union[bool, nn.Module] = True,
        postprocessing: Union[bool, nn.Module] = True,
        postprocessing_kwargs: Optional[dict] = None,
        batch_size: int = 1,
        input_image_shape: Optional[Tuple[int, int]] = None,
        input_image_channels: Optional[int] = None,
        input_image_dtype: Optional[torch.dtype] = None,
        max_predictions_per_image: Optional[int] = None,
        onnx_export_kwargs: Optional[dict] = None,
        onnx_simplify: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        output_predictions_format: DetectionOutputFormatMode = DetectionOutputFormatMode.BATCH_FORMAT,
        num_pre_nms_predictions: int = 1000,
    ):
        """
        Export the model to one of supported formats. Format is inferred from the output file extension or can be
        explicitly specified via `format` argument.

        :param output: Output file name of the exported model.
        :param nms_threshold: (float) NMS threshold for the exported model.
        :param confidence_threshold: (float) Confidence threshold for the exported model.
        :param engine: Explicit specification of the inference engine. If not specified, engine is inferred from the output file extension.
                       Supported values:
                       - "onnxruntime" - export to ONNX format with ONNX runtime as inference engine.
                       Note, models that are using NMS exported in this mode ARE compatible with TRT runtime.
                       - "tensorrt" - export to ONNX format with TensorRT  as inference engine.
                       This mode enables use of efficient TensorRT NMS plugin. Note, models that are using NMS exported in this
                       mode ARE NOT COMPATIBLE with ONNX runtime.
        :param quantization_mode: (QuantizationMode) Sets the quantization mode for the exported model.
            If None, the model is exported as-is without any changes to mode weights.
            If QuantizationMode.FP16, the model is exported with weights converted to half precision.
            If QuantizationMode.INT8, the model is exported with weights quantized to INT8. For this mode you can use calibration_loader
            to specify a data loader for calibrating the model.
        :param selective_quantizer: (SelectiveQuantizer) An optional quantizer for selectively quantizing model weights.
        :param calibration_loader: (torch.utils.data.DataLoader) An optional data loader for calibrating a quantized model.
        :param preprocessing: (bool or nn.Module)
                              If True, export a model with preprocessing that matches preprocessing params during training,
                              If False - do not use any preprocessing at all
                              If instance of nn.Module - uses given preprocessing module.
        :param postprocessing: (bool or nn.Module)
                               If True, export a model with postprocessing module obtained from model.get_post_processing_callback()
                               If False - do not use any postprocessing at all
                               If instance of nn.Module - uses given postprocessing module.
        :param postprocessing_kwargs: (dict) Optional keyword arguments for model.get_post_processing_callback(),
               used only when `postprocessing=True`.
        :param batch_size: (int) Batch size for the exported model.
        :param input_image_shape: (tuple) Input image shape (height, width) for the exported model.
               If None, the function will infer the image shape from the model's preprocessing params.
        :param input_image_channels: (int) Number of input image channels for the exported model.
               If None, the function will infer the number of channels from the model itself
               (No implemented now, will use hard-coded value of 3 for now).
        :param input_image_dtype: (torch.dtype) Type of the input image for the exported model.
                If None, the function will infer the dtype from the model's preprocessing and other parameters.
                If preprocessing is True, dtype will default to torch.uint8.
                If preprocessing is False and requested quantization mode is FP16 a torch.float16 will be used,
                otherwise a default torch.float32 dtype will be used.
        :param max_predictions_per_image: (int) Maximum number of detections per image for the exported model.
        :param device: (torch.device) Device to use for exporting the model. If not specified, the device is inferred from the model itself.
        :param onnx_export_kwargs: (dict) Optional keyword arguments for torch.onnx.export() function.
        :param onnx_simplify: (bool) If True, apply onnx-simplifier to the exported model.
        :param output_predictions_format: (DetectionOutputFormatMode) Format of the output predictions after NMS.
                Possible values:
                DetectionOutputFormatMode.BATCH_FORMAT - A tuple of 4 tensors will be returned
                (num_detections, detection_boxes, detection_scores, detection_classes)
                - A tensor of [batch_size, 1] containing the image indices for each detection.
                - A tensor of [batch_size, max_output_boxes, 4] containing the bounding box coordinates for each detection in [x1, y1, x2, y2] format.
                - A tensor of [batch_size, max_output_boxes] containing the confidence scores for each detection.
                - A tensor of [batch_size, max_output_boxes] containing the class indices for each detection.

                DetectionOutputFormatMode.FLAT_FORMAT - Tensor of shape [N, 7], where N is the total number of
                predictions in the entire batch.
                Each row will contain [image_index, x1, y1, x2, y2, class confidence, class index] values.


        :param num_pre_nms_predictions: (int) Number of predictions to keep before NMS.
        :return:
        """
        if not isinstance(self, nn.Module):
            raise TypeError(f"Export is only supported for torch.nn.Module. Got type {type(self)}")

        device: torch.device = device or infer_model_device(self)
        if device is None:
            raise ValueError(
                "Device is not specified and cannot be inferred from the model. "
                "Please specify the device explicitly: model.export(..., device=torch.device(...))"
            )

        # The following is a trick to infer the exact device index in order to make sure the model using right device.
        # User may pass device="cuda", which is not explicitly specifying device index.
        # Using this trick, we can infer the correct device (cuda:3 for instance) and use it later for checking
        # whether model places all it's parameters on the right device.
        device = torch.zeros(1).to(device).device

        logger.debug(f"Using device: {device} for exporting model {self.__class__.__name__}")

        model: nn.Module = copy.deepcopy(self).eval()

        engine: ExportTargetBackend = engine or infer_format_from_file_name(output)
        if engine is None:
            raise ValueError(
                "Export format is not specified and cannot be inferred from the output file name. "
                "Please specify the format explicitly: model.export(..., format=ExportTargetBackend.ONNXRUNTIME)"
            )

        # Infer the input image shape from the model
        if input_image_shape is None:
            input_image_shape = infer_image_shape_from_model(model)
            logger.debug(f"Inferred input image shape: {input_image_shape} from model {model.__class__.__name__}")

        if input_image_shape is None:
            raise ValueError(
                "Image shape is not specified and cannot be inferred from the model. "
                "Please specify the image shape explicitly: model.export(..., image_shape=(height, width))"
            )

        try:
            rows, cols = input_image_shape
        except ValueError:
            raise ValueError(f"Image shape must be a tuple of two integers (height, width), got {input_image_shape} instead")

        # Infer the number of input channels from the model
        if input_image_channels is None:
            input_image_channels = infer_image_input_channels(model)
            logger.debug(f"Inferred input image channels: {input_image_channels} from model {model.__class__.__name__}")

        if input_image_channels is None:
            raise ValueError(
                "Number of input channels is not specified and cannot be inferred from the model. "
                "Please specify the number of input channels explicitly: model.export(..., input_image_channels=NUM_CHANNELS_YOUR_MODEL_TAKES)"
            )

        input_shape = (batch_size, input_image_channels, rows, cols)
        prep_model_for_conversion_kwargs = {
            "input_size": input_shape,
        }

        model_type = torch.half if quantization_mode == ExportQuantizationMode.FP16 else torch.float32

        if isinstance(preprocessing, nn.Module):
            pass
        elif preprocessing is True:
            preprocessing = model.get_preprocessing_callback()
            preprocessing = nn.Sequential(CastTensorTo(model_type), preprocessing)
            input_image_dtype = input_image_dtype or torch.uint8
        else:
            preprocessing = None
            input_image_dtype = input_image_dtype or model_type

        # This variable holds the output names of the model.
        # If postprocessing is enabled, it will be set to the output names of the postprocessing module.
        output_names: Optional[List[str]] = None

        if isinstance(postprocessing, nn.Module):
            # If a user-specified postprocessing module is provided, we will attach is to the model and not
            # attempt to attach NMS step, since we do not know what the user-specified postprocessing module does,
            # and what outputs it produces.
            attach_nms_postprocessing = False
        elif postprocessing is True:
            attach_nms_postprocessing = True
            postprocessing_kwargs = postprocessing_kwargs or {}
            postprocessing_kwargs["num_pre_nms_predictions"] = num_pre_nms_predictions
            postprocessing: AbstractObjectDetectionDecodingModule = model.get_decoding_module(**postprocessing_kwargs)

            output_names = postprocessing.get_output_names()
            num_pre_nms_predictions = postprocessing.num_pre_nms_predictions
            max_predictions_per_image = max_predictions_per_image or num_pre_nms_predictions

            nms_threshold = nms_threshold or getattr(model, "_default_nms_iou", None)
            if nms_threshold is None:
                raise ValueError(
                    "nms_threshold is not specified and cannot be inferred from the model. "
                    "Please specify the nms_threshold explicitly: model.export(..., nms_threshold=0.5)"
                )

            confidence_threshold = confidence_threshold or getattr(model, "_default_nms_conf", None)
            if confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold is not specified and cannot be inferred from the model. "
                    "Please specify the confidence_threshold explicitly: model.export(..., confidence_threshold=0.5)"
                )

            if max_predictions_per_image > num_pre_nms_predictions:
                raise ValueError(
                    f"max_predictions_per_image={max_predictions_per_image} is greater than "
                    f"num_pre_nms_predictions={num_pre_nms_predictions}. "
                    f"Please specify max_predictions_per_image <= {num_pre_nms_predictions}."
                )
        else:
            attach_nms_postprocessing = False
            postprocessing = None

        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**prep_model_for_conversion_kwargs)

        # TODO: Check whether model is quantized, if yes - skip
        if quantization_mode == ExportQuantizationMode.INT8:
            from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
            from pytorch_quantization import nn as quant_nn

            patch_pytorch_quantization_modules_if_needed()

            q_util = selective_quantizer or SelectiveQuantizer(
                default_quant_modules_calibrator_weights="max",
                default_quant_modules_calibrator_inputs="histogram",
                default_per_channel_quant_weights=True,
                default_learn_amax=False,
                verbose=True,
            )
            q_util.quantize_module(model)

            if calibration_loader:
                logger.debug("Calibrating model")
                calibrator = QuantizationCalibrator(verbose=True)
                calibrator.calibrate_model(
                    model,
                    method="percentile",
                    calib_data_loader=calibration_loader,
                    num_calib_batches=16,  # TODO: How do we choose that ????
                    percentile=99.99,
                )
                logger.debug("Calibrating model complete")

        from super_gradients.training.models.conversion import ConvertableCompletePipelineModel

        # The model.prep_model_for_conversion will be called inside ConvertableCompletePipelineModel once more,
        # but as long as implementation of prep_model_for_conversion is idempotent, it should be fine.
        complete_model = (
            ConvertableCompletePipelineModel(model=model, pre_process=preprocessing, post_process=postprocessing, **prep_model_for_conversion_kwargs)
            .to(device)
            .eval()
        )

        if quantization_mode == ExportQuantizationMode.FP16:
            # For FP16 quantization, we simply can to convert the whole model to half precision
            complete_model = complete_model.half()

            if calibration_loader is not None:
                logger.warning(
                    "It seems you've passed calibration_loader to export function, but quantization_mode is set to FP16. "
                    "FP16 quantization is done by calling model.half() so you don't need to pass calibration_loader, as it will be ignored."
                )

        if engine in {ExportTargetBackend.ONNXRUNTIME, ExportTargetBackend.TENSORRT}:
            onnx_export_kwargs = onnx_export_kwargs or {}

            if quantization_mode == ExportQuantizationMode.INT8:
                use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
                quant_nn.TensorQuantizer.use_fb_fake_quant = True

            try:
                with torch.no_grad():
                    onnx_input = torch.randn(input_shape).to(device=device, dtype=input_image_dtype)
                    # Sanity check that model works
                    _ = complete_model(onnx_input)

                    for name, p in complete_model.named_parameters():
                        if p.device != device:
                            logger.warning(f"Model parameter {name} is on device {p.device} but expected to be on device {device}")

                    for name, p in complete_model.named_buffers():
                        if p.device != device:
                            logger.warning(f"Model buffer {name} is on device {p.device} but expected to be on device {device}")

                    logger.debug("Exporting model to ONNX")
                    logger.debug(f"ONNX input shape: {input_shape} with dtype: {input_image_dtype}")
                    logger.debug(f"ONNX output names: {output_names}")
                    logger.debug(f"ONNX export kwargs: {onnx_export_kwargs}")

                    if onnx_export_kwargs is not None and "args" in onnx_export_kwargs:
                        raise ValueError(
                            "`args` key found in onnx_export_kwargs. We explicitly construct dummy input (`args`) inside export() method. "
                            "Overriding args is not supported so please remove it from the `onnx_export_kwargs`."
                        )
                    torch.onnx.export(model=complete_model, args=onnx_input, f=output, output_names=output_names, **onnx_export_kwargs)

                # Stitch ONNX graph with NMS postprocessing
                if attach_nms_postprocessing:
                    if engine == ExportTargetBackend.TENSORRT:

                        if onnx_simplify:
                            # If TRT engine is used, we need to run onnxsim.simplify BEFORE attaching NMS,
                            # because EfficientNMS_TRT is not supported by onnxsim and would lead to a runtime error.
                            onnxsim.simplify(output)
                            logger.debug(f"Ran onnxsim.simplify on model {output}")
                            # Disable onnx_simplify to avoid running it twice.
                            onnx_simplify = False

                        nms_attach_method = attach_tensorrt_nms

                        if output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
                            logger.warning(
                                "Support of flat predictions format in TensorRT is experimental and may not work on all versions of TensorRT. "
                                "We recommend using TensorRT 8.4.1 or newer. On older versions this format will not work. "
                                "If you encountering issues loading exported model in TensorRT, please try upgrading TensorRT to latest version. "
                                "Alternatively, you can export the model to output predictions in batch format by "
                                "specifying output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT. "
                            )
                    elif engine == ExportTargetBackend.ONNXRUNTIME:
                        nms_attach_method = attach_onnx_nms
                    else:
                        raise KeyError(f"Unsupported engine: {engine}")

                    nms_attach_method(
                        onnx_model_path=output,
                        output_onnx_model_path=output,
                        num_pre_nms_predictions=num_pre_nms_predictions,
                        max_predictions_per_image=max_predictions_per_image,
                        nms_threshold=nms_threshold,
                        confidence_threshold=confidence_threshold,
                        batch_size=batch_size,
                        output_predictions_format=output_predictions_format,
                        device=device,
                    )

                if onnx_simplify:
                    onnxsim.simplify(output)
                    logger.debug(f"Ran onnxsim.simplify on {output}")
            finally:
                if quantization_mode == ExportQuantizationMode.INT8:
                    # Restore functions of quant_nn back as expected
                    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state

        else:
            raise ValueError(f"Unsupported export format: {engine}. Supported formats: onnxruntime, tensorrt")

        # Cleanup memory, not sure whether it is necessary but just in case
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ModelExportResult(
            input_image_channels=input_image_channels,
            input_image_dtype=input_image_dtype,
            input_image_shape=input_image_shape,
            engine=engine,
            quantization_mode=quantization_mode,
            output=output,
            output_predictions_format=output_predictions_format,
        )
