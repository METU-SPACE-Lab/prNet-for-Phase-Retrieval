# [prNet: Data-Driven Phase Retrieval via Stochastic Refinement](https://arxiv.org/abs/2507.09608)
Mehmet Onurcan Kaya and Figen S. Oktem

This repository contains the official codes for the paper "[prNet: Data-Driven Phase Retrieval via Stochastic Refinement](https://arxiv.org/abs/2507.09608)".

## Abstract
We propose a novel framework for phase retrieval that leverages Langevin dynamics to enable efficient posterior sampling, yielding reconstructions that explicitly balance distortion and perceptual quality. Unlike conventional approaches that prioritize pixel-wise accuracy, our method navigates the perception-distortion tradeoff through a principled combination of stochastic sampling, learned denoising, and model-based updates. The framework comprises three variants of increasing complexity, integrating theoretically grounded Langevin inference, adaptive noise schedule learning, parallel reconstruction sampling, and warm-start initialization from classical solvers. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple benchmarks, both in terms of fidelity and perceptual quality.

## Getting Started

### Main files:

https://github.com/METU-SPACE-Lab/prnet/tree/main/notebooks

### Evaluation results:

https://github.com/METU-SPACE-Lab/prnet/tree/main/logs


### Model weights:

prNet-Large, Main Loop Denoiser: https://terabox.com/s/1GFeV9kMiWAX36Nrh90gfEQ

prNet-Large-Adversarial, Final Denoiser: https://terabox.com/s/1G9m9fKVMlr3B2_mW1Z7juw

prNet-Small: https://terabox.com/s/1uI9xpU0QBepu9zfyHFcKsw


## Citation
Please cite the following paper when using this code or data:
```
@misc{kaya2025prnetdatadrivenphaseretrieval,
      title={prNet: Data-Driven Phase Retrieval via Stochastic Refinement}, 
      author={Mehmet Onurcan Kaya and Figen S. Oktem},
      year={2025},
      eprint={2507.09608},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2507.09608}, 
}
```

Short conference paper:
```
@inproceedings{kaya2025prnet,
  title={prNet: Efficient and Robust Phase Retrieval via Stochastic Refinement},
  author={Kaya, Mehmet Onurcan and Oktem, Figen S},
  booktitle={2025 IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={01--06},
  year={2025},
  organization={IEEE}
}
```

## Contact
If you have any questions or need help, please feel free to contact me via monka@dtu.dk.




