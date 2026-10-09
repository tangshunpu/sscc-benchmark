# SSCC Benchmark：信源编码 + 信道编码

这是从原 `jscc_baseline` 和 `image_compression` 项目中整理出的独立仓库。
支持 **BPG / VTM / MS-ILLM / ELiC + 5G LDPC + QAM + AWGN**，可对 Kodak、CLIC
数据集生成真实压缩码流并运行 benchmark。运行不依赖原项目路径。

## 安装和快速验证

建议 Linux x86-64、Python 3.12。在本仓库根目录执行：

```bash
uv sync --locked --extra channel --extra dev
source .venv/bin/activate

# 如主机尚未安装构建依赖，先执行：
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl yasm pkg-config \
  libpng-dev libjpeg-dev zlib1g-dev libssl-dev

bash scripts/install_bpg.sh
bash scripts/install_vtm.sh
export PATH="$PWD/.local/bin:$PATH"
```

[BPG/VTM 上游仓库和完整安装说明](docs/codecs.md)。脚本在 `third_party/` 编译，
安装到 `.local/bin/`。也可使用自己的二进制，通过 `--bpg-encoder`、`--bpg-decoder`、
`--vtm-encoder`、`--vtm-decoder`、`--vtm-config` 指定。

把 [Kodak](https://r0k.us/graphics/kodak/) 的 24 张 PNG 放入 `data/kodak/`；
CLIC 需从[数据集官网](https://www.compression.cc/)选择年份、划分后下载。
示例配置使用 CLIC2020 test 的 `mobile_test` 和 `professional_test`。
仓库不附带数据集，已有数据可以直接指定路径，无需复制。

```bash
# 先验证一张 Kodak 图
sscc-prepare --method BPG --dataset kodak --dataset-root data/kodak \
  --qualities 40 48 51 --max-images 1
sscc-benchmark --method BPG --dataset kodak --dataset-root data/kodak \
  --snr-db 20 --repeats 1 --max-images 1 --device cpu --metrics psnr
```

## 批量 benchmark

```bash
for method in BPG VTM; do
  sscc-prepare --method "$method" --dataset kodak --dataset-root data/kodak
  sscc-prepare --method "$method" --dataset clic2020_test \
    --dataset-root data/clic2020/mobile_test \
    --dataset-root data/clic2020/professional_test
done
sscc-suite examples/kodak-clic.toml --dry-run
sscc-suite examples/kodak-clic.toml
sscc-summarize results --output benchmark.csv
```

默认压缩率 CR=96，8 个 SNR 点（0、3、6、9、12、15、18、20 dB），重复 3 次。
BPG/VTM 码流生成支持基于输入、配置、输出哈希的续跑；学习式码流会重新生成，
防止换权重后误用旧缓存。benchmark 默认拒绝覆盖已有实验。
VTM 编码较慢，建议先用 `--max-images 1` 检查环境。

学习式方法需要权重和上游源码；[MS-ILLM/ELiC 安装和运行步骤](docs/learned-codecs.md)。
Sionna 与 CompressAI 的 NumPy 要求冲突，采用两个环境，通过 `--codec-python`
调用学习式解码器，不需要强制覆盖依赖。ELiC 使用真实模型解码，不读取原项目的重建缓存。

## 输出与实验口径

每个 SNR 输出 `run.json`、逐图 CSV、平均指标 CSV 和重建图；可追踪输入文件哈希、
依赖版本、码率、实际信道符号数、BER、解码成功率、fallback 比例。
默认只算 PSNR；安装 `metrics` extra 后可选择 MS-SSIM、LPIPS、DISTS、PieAPP、FID、KID。

信道预算为 `floor(3*H*W/CR)` 个复符号，完整容器、32-bit 长度头和 padding 均计入预算。
SNR 指 Es/N0；SNR 到 MCS 的选择是带 3 dB 裕量的启发式映射。
为兼容原实验，失败时使用原图逐通道均值图；这是使用原图信息的 oracle fallback，
均值信息未计入传输预算，论文或报告必须同时说明该约定和失败比例。

## 引用

如果本仓库对你的研究有帮助，欢迎引用我们的论文：

[Enabling Training-Free Semantic Communication Systems with Generative Diffusion Models](https://doi.org/10.1109/GLOBECOM59602.2025.11431826)，IEEE GLOBECOM 2025。

```bibtex
@inproceedings{tang2025trainingfree,
  title     = {Enabling Training-Free Semantic Communication Systems with Generative Diffusion Models},
  author    = {Tang, Shunpu and Jia, Yuanyuan and Yang, Qianqian and Zhang, Ruichen and Park, Jihong and Niyato, Dusit},
  booktitle = {2025 IEEE Global Communications Conference (GLOBECOM)},
  year      = {2025},
  doi       = {10.1109/GLOBECOM59602.2025.11431826},
  url       = {https://doi.org/10.1109/GLOBECOM59602.2025.11431826}
}
```

[BibTeX 文件](citation.bib)

[完整 README](README.md) · [实验协议](docs/benchmark.md) ·
[原项目与上游链接](docs/upstream.md) · [验证记录](docs/validation.md)
