# Third-party notices

## Apple AXLearn transducer alignment algorithm

Portions of `src/fcrnnt_jax/rnnt_loss.py`—specifically the diagonal lattice
layout, forward/backward recurrences, padding construction, and analytic VJP—are
adapted and substantially modified from
[`axlearn/common/transducer.py`](https://github.com/apple/axlearn/blob/main/axlearn/common/transducer.py).

Copyright © 2023 Apple Inc.

AXLearn is licensed under the Apache License, Version 2.0. A copy of the
license is included in this repository and is available in the AXLearn repository at
<https://github.com/apple/axlearn/blob/main/LICENSE> and at
<https://www.apache.org/licenses/LICENSE-2.0>.

The fcrnnt-jax version replaces AXLearn's layer/config dependencies with a
standalone API, adds explicit length handling and NeMo-compatible reductions,
forces FP32 loss mathematics, supports empty targets, and adds a frame-streamed
joint-network interface.
