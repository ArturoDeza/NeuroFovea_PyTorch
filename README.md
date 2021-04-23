<img src="https://github.com/ArturoDeza/NeuroFovea_PyTorch/blob/main/Metamer_Transform_Update.png" width="900">



# NeuroFovea_PyTorch
An adapted version of the Metamer Foveation Transform code from Deza et al. ICLR 2019


To complete the installation please run:

```
$ bash download_models_and_stimuli.sh
```

### Example code:

Generate a foveated image for the `512x512` image `1.png` with a center fixation, specified by the rate of growth of the receptive field: `s=0.4`. Note: The approximate rendering time for a metamer should be around 1 second if you have your GPU on.

```
$ python Metamer_Transform.py --image 1.png --scale 0.4 --reference 0
```

The paper "Emergent Properties of Foveated Perceptual Systems" of Deza & Konkle, 2020/2021 (https://arxiv.org/abs/2006.07991) that uses a foveated transform (with an exagerated distortion given the scaling factor set to `s=0.4`) was ran with the lua code accessible here: https://github.com/ArturoDeza/NeuroFovea, but current and future follow-up work has transitioned to this PyTorch version. After finally vetting the code (and making sure both the lua + PyTorch versions produce the same outputs), we've decide to release it to accelerate work on spatially-adaptive (foveated) texture computation in humans and machines.

The Foveated Texture Transform essentially computes log-polar + localized Adaptive Instance Normalization (See Huang & Belongie (ICCV, 2019); This code is thus an extension of: https://github.com/naoto0804/pytorch-AdaIN)

Please read our paper to learn more about visual metamerism: https://openreview.net/forum?id=BJzbG20cFQ

We hope this code and our paper can help researchers, scientists and engineers improve the use and design of metamer models that have potentially exciting applications in both computer vision and visual neuroscience.

This code is free to use for Research Purposes, and if used/modified in any way please consider citing:

```
@inproceedings{
deza2018towards,
title={Towards Metamerism via Foveated Style Transfer},
author={Arturo Deza and Aditya Jonnalagadda and Miguel P. Eckstein},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=BJzbG20cFQ},
}
```

Other inquiries: deza@mit.edu
