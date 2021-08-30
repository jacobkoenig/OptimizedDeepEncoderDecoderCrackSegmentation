# Optimized Deep Encoder-Decoder Methods for Crack Segmentation
---
This repository contains the code implementation of the main model from our paper *Optimized Deep Encoder-Decoder Methods for Crack Segmentation* which uses an EfficientNet B5 backbone. 


König, J., Jenkins, M.D., Mannion, M., Barrie, P. and Morison, G., 2021. Optimized deep encoder-decoder methods for crack segmentation. Digital Signal Processing, 108, p.102907. [[Paper]](https://www.sciencedirect.com/science/article/pii/S1051200420302529)

#### Abstract
- Surface crack segmentation poses a challenging computer vision task as background, shape, color and size of cracks vary. In this work we propose optimized deep encoder-decoder methods consisting of a combination of techniques which yield an increase in crack segmentation performance. Specifically we propose a decoder-part for an encoder-decoder based deep learning architecture for semantic segmentation and study its components to achieve increased performance. We also examine the use of different encoder strategies and introduce a data augmentation policy to increase the amount of available training data. The performance evaluation of our method is carried out on four publicly available crack segmentation datasets. Additionally, we introduce two techniques into the field of surface crack segmentation, previously not used there: Generating results using test-time-augmentation and performing a statistical result analysis over multiple training runs. The former approach generally yields increased performance results, whereas the latter allows for more reproducible and better representability of a methods results. Using those aforementioned strategies with our proposed encoder-decoder architecture we are able to achieve new state of the art results in all datasets.

---

The models in this paper have been trained/evaluated on [CrackForest](https://github.com/cuilimeng/CrackForest-dataset), the [DeepCrack (Transactions on Image Processing)](https://ieeexplore.ieee.org/document/8517148) and [Deepcrack (Neurocomputing)](https://github.com/yhlleo/DeepCrack) datasets.


# Code 
The code for the model is contained within the `model.py` file. 
To generate the model for training call the `get_model()` function. 

# Requirements
This code requries Tensorflow 2.0 or higher and the EfficientNets from [here](https://github.com/qubvel/efficientnet).

--- 
# Reference
If you uses our proposed model or code please cite our paper:

```
@article{konig2021optimized,
  title={Optimized deep encoder-decoder methods for crack segmentation},
  author={K{\"o}nig, Jacob and Jenkins, Mark David and Mannion, Mike and Barrie, Peter and Morison, Gordon},
  journal={Digital Signal Processing},
  volume={108},
  pages={102907},
  year={2021},
  publisher={Elsevier}
}
```

# Acknowledgements
We thank [[@qubvel](https://github.com/qubvel)] for his EfficientNet implementation from [here](https://github.com/qubvel/efficientnet) which we have used in this work.
