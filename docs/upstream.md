# Project origins and upstream implementations

This standalone repository was extracted from the user's local `jscc_baseline`
project (digital channel, adapters, dataset matching, metrics and benchmark) and
`image_compression` project (source encoder wrappers). Neither local project
provided a public Git remote in this extraction; no public origin URL is invented.
The original projects remain untouched. `SOURCE_ORIGINS.json` records paths
relative to those original projects and the SHA-256 of files before adaptation.
There is no runtime dependency on either original checkout.

| Component | Original project/source | Role |
|---|---|---|
| BPG | https://bellard.org/bpg/ | Original libbpg 0.9.8 release, encoder and decoder |
| BPG mirror | https://github.com/mirrorer/libbpg | Unofficial source mirror, not the release authority |
| VTM | https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM | JVET VVC reference encoder/decoder |
| Sionna | https://github.com/NVlabs/sionna | LDPC, QAM, AWGN, soft demapper |
| MS-ILLM | https://github.com/facebookresearch/NeuralCompression/tree/main/projects/illm | Learned source codec |
| ELiC reimplementation | https://github.com/VincentChandelier/ELiC-ReImplemetation | Learned source codec network and checkpoint instructions |
| CompressAI | https://github.com/InterDigitalInc/CompressAI | Entropy coding/model support for learned codecs |
| IQA-PyTorch | https://github.com/chaofengc/IQA-PyTorch | Optional learned pair metrics |
| pytorch-msssim | https://github.com/VainF/pytorch-msssim | MS-SSIM |
| torch-fidelity | https://github.com/toshas/torch-fidelity | FID/KID |
| Kodak | https://r0k.us/graphics/kodak/ | Lossless evaluation images |
| CLIC | https://www.compression.cc/ | Challenge dataset information |

References:

- [3GPP TS 38.214](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3216)
  for MCS Table 5.1.3.1-1. Our SNR mapping is an explicitly documented heuristic.
- [MS-ILLM paper](https://proceedings.mlr.press/v202/muckley23a.html).
- [ELIC paper](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_Adaptive_CVPR_2022_paper.html).

The public harness does not vendor any upstream codec source, binaries, weights
or datasets. Installers and documentation point to the original projects.
