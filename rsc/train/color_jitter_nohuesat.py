import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import (adjust_brightness_accumulative,
                            adjust_contrast_with_mean_subtraction)


class ColorJitterNoHueSat(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/ColorJitter.png

    Args:
        p: probability of applying the transformation.
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        silence_instantiation_warning: if True, silence the warning at instantiation.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`
    """

    def __init__(
        self,
        brightness: Union[Tensor, float, Tuple[float, float],
                          List[float]] = 0.0,
        contrast: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        silence_instantiation_warning: bool = False,
    ) -> None:
        super().__init__(p=p,
                         same_on_batch=same_on_batch,
                         keepdim=keepdim)

        if not silence_instantiation_warning:
            warnings.warn(
                "`ColorJitter` is now following Torchvision implementation. Old "
                "behavior can be retrieved by instantiating `ColorJiggle`.",
                category=DeprecationWarning,
            )

        self.brightness = brightness
        self.contrast = contrast
        self._param_generator = cast(
            rg.ColorJitterGenerator,
            rg.ColorJitterGenerator(brightness, contrast))

    def apply_transform(self,
                        input: Tensor,
                        params: Dict[str, Tensor],
                        flags: Dict[str, Any],
                        transform: Optional[Tensor] = None) -> Tensor:

        transforms = [
            lambda img: adjust_brightness_accumulative(
                img, params["brightness_factor"]),
            lambda img: adjust_contrast_with_mean_subtraction(
                img, params["contrast_factor"])
        ]

        jittered = input
        for idx in (0, 1):
            t = transforms[idx]
            jittered = t(jittered)

        return jittered
