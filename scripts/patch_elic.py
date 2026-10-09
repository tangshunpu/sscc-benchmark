"""Apply the two import compatibility edits needed by ELiC + CompressAI 1.2.8."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("elic_root", type=Path)
args = p.parse_args()
path = args.elic_root / "Network.py"
text = path.read_text()
text = text.replace("from compressai.models.priors import CompressionModel, GaussianConditional",
                    "from compressai.models.priors import CompressionModel\nfrom compressai.entropy_models import GaussianConditional")
text = text.replace("from compressai.ops import ste_round",
                    "try:\n    from compressai.ops import ste_round\nexcept ImportError:\n    def ste_round(x):\n        return torch.round(x) - x.detach() + x")
if text != path.read_text():
    path.write_text(text)
    print(f"Patched {path}")
else:
    print(f"No changes needed in {path}")
