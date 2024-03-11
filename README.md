# HTE_PINN
Here is the official JAX implementation of the paper **Hutchinson Trace Estimation for High-Dimensional and High-Order Physics-Informed Neural Networks**.

The paper is coauthored by **Zheyuan Hu, Zekun Shi, George Em Karniadakis, Kenji Kawaguchi**.

It has been accepted by **Computer Methods in Applied Mechanics and Engineering**.

arXiv version: https://arxiv.org/abs/2312.14499

Journal version (Open Access): https://www.sciencedirect.com/science/article/pii/S0045782524001397

# Code Explanation

Gordon1.py: Code for the high-dimensional second-order Sine-Gordon equation with a two-body solution, which includes both PINN and gradient-enhanced PINN (gPINN).

Gordon2.py: Code for the high-dimensional second-order Sine-Gordon equation with a three-body solution, which includes both PINN and gradient-enhanced PINN (gPINN).

Biharmonic.py: Code for the high-dimensional fourth-order biharmonic equation.

# Citations

If you think the code is useful, kindly cite our paper.

```bibtex
@article{hu2024hutchinson,
  title={Hutchinson trace estimation for high-dimensional and high-order physics-informed neural networks},
  author={Hu, Zheyuan and Shi, Zekun and Karniadakis, George Em and Kawaguchi, Kenji},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={424},
  pages={116883},
  year={2024},
  publisher={Elsevier}
}
```

You may also consider citing our other papers on high-dimensional and high-order PINN and PDE:

```bibtex
@article{hu2023tackling,
  title={Tackling the curse of dimensionality with physics-informed neural networks},
  author={Hu, Zheyuan and Shukla, Khemraj and Karniadakis, George Em and Kawaguchi, Kenji},
  journal={arXiv preprint arXiv:2307.12306},
  year={2023}
}
```

```bibtex
@article{hu2023bias,
  title={Bias-variance trade-off in physics-informed neural networks with randomized smoothing for high-dimensional PDEs},
  author={Hu, Zheyuan and Yang, Zhouhao and Wang, Yezhen and Karniadakis, George Em and Kawaguchi, Kenji},
  journal={arXiv preprint arXiv:2311.15283},
  year={2023}
}
```


```bibtex
@article{hu2024score,
  title={Score-Based Physics-Informed Neural Networks for High-Dimensional Fokker-Planck Equations},
  author={Hu, Zheyuan and Zhang, Zhongqiang and Karniadakis, George Em and Kawaguchi, Kenji},
  journal={arXiv preprint arXiv:2402.07465},
  year={2024}
}
```
