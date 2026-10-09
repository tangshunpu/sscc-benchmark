# Third-party notices

The MIT license in this repository applies to this Python harness and its
project-owned adapters. It does not relicense external software or assets.
No external codec binaries, source checkouts, model weights or dataset images
are bundled in the source distribution.

- BPG: see https://bellard.org/bpg/ and the licenses in its release archive.
  The decoder combines BSD-licensed library code and LGPL 2.1 FFmpeg components;
  the encoder as a whole is distributed under GPL 2. Read component licenses
  before redistributing compiled binaries.
- VTM: see the `COPYING` file of the JVET reference software checkout.
- Sionna: see the NVIDIA upstream license.
- NeuralCompression: its repository identifies the code as MIT and released
  weights as CC-BY-NC 4.0. Do not treat the weights as MIT-licensed.
- ELiC: see the reimplementation's license and checkpoint conditions. Model
  assets must be obtained by the user; this repository does not claim ownership.
- CompressAI, PyTorch, Pillow, metric packages: retain their respective licenses.
- Kodak and CLIC: obtain from their providers and follow their dataset terms.

Links are collected in [docs/upstream.md](docs/upstream.md). Distribution of this
harness does not imply permission to redistribute all referenced datasets or
weights. The native codecs are invoked as external programs, not linked into
our Python wheel.
