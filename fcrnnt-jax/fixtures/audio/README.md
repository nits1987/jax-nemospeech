# Parakeet real-audio smoke fixture

The audio is NVIDIA's example for `nvidia/parakeet-rnnt-1.1b` and is not stored
in this repository. Download and verify the exact 237,964-byte PCM16 mono,
16-kHz WAV:

```bash
curl -fL --retry 3 \
  -o fixtures/audio/2086-149220-0033.wav \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav

echo '5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150  fixtures/audio/2086-149220-0033.wav' \
  | sha256sum -c -
```

The reference transcript is in `2086-149220-0033.txt`. The utterance is
LibriSpeech `dev-clean` item `2086-149220-0033`; see
`THIRD_PARTY_NOTICES.md` for attribution.

`audio-smoke` verifies file identity, WAV decoding, and JAX log-mel extraction.
It deliberately does not emit an ASR hypothesis or WER because the PoC does not
yet contain converted Parakeet weights, the matching tokenizer, or an RNN-T
decoder.
