
from .bpg import BPGDecoder
from .vtm import VTMDecoder


def create_decoder(method, device="auto", bpg_decoder="bpgdec", vtm_decoder="DecoderApp",
                   msillm_torch_hub_repo=None, elic_root=None, elic_checkpoint_dir=None,
                   codec_python=None):
    key = method.casefold().replace("-", "")
    if key == "bpg":
        return BPGDecoder(bpg_decoder)
    if key == "vtm":
        return VTMDecoder(vtm_decoder)
    if codec_python:
        from .worker import ProcessDecoder
        return ProcessDecoder(codec_python, method, device, msillm_torch_hub_repo,
                              elic_root, elic_checkpoint_dir)
    if key == "msillm":
        from .msillm import MSILLMDecoder
        return MSILLMDecoder(device, msillm_torch_hub_repo)
    if key == "elic":
        from .elic import ELiCDecoder
        return ELiCDecoder(device, elic_root, elic_checkpoint_dir)
    raise ValueError(f"Unsupported method: {method}")
