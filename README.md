# Learning to Generate Human-Human-Object Interactions from Textual Descriptions

## [Project Page](https://tlb-miss.github.io/hhoi/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2511.20446) 

## Installation

### System Requirements

This code has been tested in the following settings, but is expected to work in other systems. 
- Ubuntu 20.04
- CUDA 11.8
- NVIDIA RTX A6000

### Conda Environment
``` bash
conda create -n hhoi python=3.8
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout -f v0.7.2
# Change "-std=c++14" to "-std=c++17" in setup.py
pip install -e .
pip install opencv-python==4.2.0.32 scipy==1.4.1 numpy==1.23.5 tensorboardX==2.5.1 diffusers==0.35.2 datasets trimesh[easy] openai-clip wandb matplotlib smplx torchdiffeq cvxpy
```

## Dataset and Model Preparation

### SMPL-X

After downloading ``smplx`` from [here](https://smpl-x.is.tue.mpg.de/), organize it as follows:
``` bash
smplx
├── SMPLX_FEMALE.npz
├── SMPLX_FEMALE.pkl
├── SMPLX_MALE.npz
├── SMPLX_MALE.pkl
├── SMPLX_NEUTRAL.npz
├── SMPLX_NEUTRAL.pkl
├── SMPLX_NEUTRAL_2020.npz
```

### Training Dataset

Download the training dataset from [here](https://drive.google.com/drive/folders/10qGvmpjIDLl18fuCbxHoD9hw5g0hwxrh?usp=sharing), unzip it, and organize ``data`` directory as follows:
``` bash
data
├── hoi
      ├── bench_sit
            ├── 0000.pkl
            ├── 0001.pkl
            ...
            ├── asset.obj
            └── text_prompt.txt
      ├── board_carry
      ├── board_carry
      ...
├── hhi
      ├── bench_sit
      ...
```

### Encoder/Decoder for Human Body Pose

Download ``encoder_decoder_latest.pth`` from [here](https://drive.google.com/drive/folders/10qGvmpjIDLl18fuCbxHoD9hw5g0hwxrh?usp=sharing) and organize the ``enc_dec_human_pose`` directory as follows:
``` bash
enc_dec_human_pose
└── encoder_decoder_latest.pth
```

## Training

### Pre-trained HHOI Models

You can download pre-trained versions of hoi diffusion and hhi diffusion from [here](https://drive.google.com/drive/folders/10qGvmpjIDLl18fuCbxHoD9hw5g0hwxrh?usp=sharing). You can also train them from scratch, as shown below.

### Training HOI / HHI Diffusion

Set ``data_path`` and ``log_dir`` in ``scripts/train_hoi.sh`` (or ``scripts/train_hhi.sh``). Then,
``` bash
bash scripts/train_hoi.sh
bash scripts/train_hhi.sh
```

During training, checkpoints will be saved in the ``results/ckpts/{log_dir}`` directory.

## Inference

### HOI / HHI

Set ``log_dir``, ``score_model_path`` and ``input_text_prompt`` in ``scripts/infer_hoi.sh`` (or ``scripts/infer_hhi.sh``). Then,
``` bash
bash scripts/infer_hoi.sh
bash scripts/infer_hhi.sh
```
The inference results will be saved as follows: ``results/inference/{log_dir}/{input_text_prompt}/inference.npz``


### HHOI
First, we need to create an input pickle file.
``` bash
python3 make_hhoi_pickle.py --num_humans {number} --hoi_texts {text list} --hhi_pairs {human pair list} --hhi_texts {text list} --out_file_name {file name}
```
For example,
``` bash
python3 make_hhoi_pickle.py --num_humans 3 --hoi_texts "A person is sitting on a bench." "A person is sitting on a bench." "A person is sitting on a bench." --hhi_pairs "(1,2)" "(2,3)" --hhi_texts "Two people are sitting side by side on a bench." "Two people are sitting side by side on a bench." --out_file_name bench_3_people
```
**Note:**
- `len(hoi_texts)` == `num_humans`
- `len(hhi_texts)` == `len(hhi_pairs)`
- `hhi_pairs` must be a DAG (No need to be connected graph).

The results will be saved as follows: ``data/pickle/{out_file_name}.pkl``

Now, proceed with sampling based on the information in the pickle file.

Set ``log_dir``, ``hoi_score_model_path``, ``hhi_score_model_path`` and ``input_pickle_path`` in ``scripts/infer_hhoi.sh``. Then,
``` bash
bash scripts/infer_hhoi.sh
```
**Some args in ``infer_hhoi.sh``:**
- `use_inconsistency_loss` and `use_collision_loss`: Whether to use inconsistency loss and collision loss. Note that we cannot use collision loss alone for now.
- `time_thres`: Whether to apply the collision loss and inconsistency loss starting from a specific time step within the range `[1, eps]`
- `solver_type` == `scipy`'s ode solver (cpu) or `torchdiffeq` (gpu)
- `additional_loss_step`: How frequently to apply the inconsistency loss and collision loss during the integrator steps. A higher number leads to faster sampling but potentially lower accuracy.
- `sampling_steps_for_gpu_solver`: If `solver_type` is set to `gpu`, determines how many intermediate results the solver returns during integration. Unless you need intermediate results, keep the default value of `2`. Increasing it may slow down sampling.

The inference results will be saved as follows: ``results/inference/{log_dir}/{pickle_file_name}/inference.npz``

## Visualize

Use ``get_mesh.py`` or ``get_mesh2.py``. The former exports objects and all humans as a single obj file, while the latter exports each object and human separately.

``` bash
python3 get_mesh.py --data_dir {data_dir} --inference_dir {inference_dir}
python3 get_mesh2.py --data_dir {data_dir} --inference_dir {inference_dir}
```
For example,
``` bash
python3 get_mesh.py --data_dir data/hoi/bench_sit --inference_dir results/inference/HHOI/bench_3_people
python3 get_mesh2.py --data_dir data/hoi/bench_sit --inference_dir results/inference/HHOI/bench_3_people
```
``{inference_dir}`` must be the directory path where ``inference.npz`` is located.

The results of ``get_mesh.py`` are saved in the ``{inference_dir}/mesh_outputs`` directory, and the results of ``get_mesh2.py`` are saved in the ``{inference_dir}/mesh_outputs_per_obj`` directory.

## Post-processing (Optional)

We can perform post-processing to slightly alleviate interpenetration between meshes. However, it is unlikely to yield a significant gain.

``` bash
python3 post_process.py --target_folder results/inference/HHOI/bench_3_people/mesh_outputs_per_obj/0
```

The results are saved in ``results/inference/HHOI/bench_3_people/mesh_outputs_per_obj_opt/0`` when you execute the above code.

## Citation
```bibtex
@inproceedings{hhoi,
      title={Learning to Generate Human-Human-Object Interactions from Textual Descriptions}, 
      author={Na, Jeonghyeon and Baik, Sangwon and Lee, Inhee and Lee, Junyoung and Joo, Hanbyul},
      booktitle={NeurIPS},
      year={2025}
}
```
