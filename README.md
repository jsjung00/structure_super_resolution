# Heat kernel diffusion for 3D structure super resolution
Note: Code adadpted from e3_diffusion_for_molecules.

## Introduction
A fundamental limitation to analyzing and understanding molecular structures is the resolution under which the molecular structure is imaged. For some experimental datasets, the microscopy imaging process is noisy and the imaging equipment poses a fundamental signal bottleneck, preventing detailed understanding of the molecular structure. To resolve this, we propose a heat kernel based denoising diffusion model that learns to super-resolve 3D molecular structures. Our motivation is to exploit high resolution molecular structures as a prior that we can learn an accurate forward process from (i.e high resolution to low resolution). Then, we train a denoising diffusion model to reverse this process, allowing us to super-resolve a low resolution structure to a high resolution one. Drawing from scale space theory, we utilize the heat equation as a nice forward process that progressively removes higher frequency details. We adapt the heat equation and define our forward process as spatial gaussian convolutions of increasing radius. 

## Method
In order to learn a low to high resolution reverse process, we first define a high to low resolution forward process. Motivated by the heat equation, we parameterize this by spatial gaussian convolution of increasing radius.

Specifically, given a set of atomic or node coordinates $X_0 \in \mathbb{R}^{n \times 3}$, to define our forward process with some time $t \in [0,1]$, we utilize a gaussian heat kernel $G_t$ of increasing variance. We define our corrupted data $X_t$ as the convolution $X_t = G_t * X_0 = D(X_0, t)$. 

We visualize the forward process below with some example 3D structures that show progressively increasing gaussian blurring. 

To learn the reverse process, we utilize a denoising diffusion model framework as in "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise", (Bansal et al. 2022). Specifically we define our loss as the denoising reconstruction objection 
$E_{t\in U[0,1], x_0 \in X}[||R_\theta (D(x_0,t), t) - x_0||]$ where $R_\theta$ is a learned denoising diffusion model.

We adopt the same E(3) equivariant diffusion model as in "Equivariant Diffusion for Molecule Generation in 3D" (Hoogeboom et al. 2022); as our backbone we use a equivariant graph neural network (EGNN) to learn E(3) equivariant node coordinates. 

## Preliminary experiments
We run some preliminary experiments on the larger scale molecular conformer dataset GEOM (Axelrod & Gomez-Bombarelli, 2020). We think that this is reasonable dataset as it contains molecules with up 181 atoms and 44.4 atoms on average.


### For GEOM-Drugs

Training
```python main_geom_drugs.py --n_epochs 3000 --exp_name edm_geom_drugs --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000```



