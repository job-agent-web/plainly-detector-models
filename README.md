# Public browser detector model distribution

Open-source model components used by Plainly. This site distributes only model files and attribution. It collects no user text, credentials or payments and provides no hosted inference service.

Weights are pinned, converted with ONNX Runtime 1.23.2, and checked against original model outputs before release. Weight-only int8 retains float activation precision. This is a conversion parity check, not an accuracy evaluation. Estimates are not proof of authorship.

- slop: https://huggingface.co/anudit/slop
  - Licence: MIT. See upstream source and accompanying notices.
  - SHA256: 7c423ca216d6cc3a1be13d69f37e6f263da5bf8a51ac2b23e8de51cad54e5268
- smollm: https://huggingface.co/Joshfcooper/ai-text-detector-optimized
  - Licence: Apache-2.0. See upstream source and accompanying notices.
  - SHA256: 1fc9cecce4d1e8b5496a544170939a06c6a33dce80640dd4b590c8855b8447f7

SLOP's FP16 operations were promoted to FP32 for CPU support; its released q4 weights are retained. SLOP requires 17 readability features. Plainly uses standard English formulas and a syllable heuristic because the author's feature implementation is unavailable. SmolLM's output named probability_human is documented as inverted; its raw output is interpreted as the AI estimate.

First-time downloads are large. Browser caches may be evicted; model inference needs a desktop browser with sufficient memory.
