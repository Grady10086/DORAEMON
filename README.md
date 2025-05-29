<h1 align="center">🔔 DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation</a>
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2505.21969"><img src="https://img.shields.io/badge/arXiv-2505.21969-b31b1b.svg" alt="Paper"></a>
  <a href="https://grady10086.github.io/DORAEMON/"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
  <a href="#-citation"><img src="https://img.shields.io/badge/Citation-BibTeX-blue.svg" alt="Citation"></a>
</p>

## 📚 Contents
- [Abstract](#Abstract)
- [Update](#Update)
- [Demo](#Demo)
- [Get Started](#Get-Started)
- [Evaluation](#Evaluation)
- [Citation](#Citation)

## ✨ Abstract
Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develop Nav-Ensurance to ensure navigation safety and efficiency.

## 💥 Update
🔥 We've reorganized and cleaned up the repository to ensure a clear, well-structured codebase. Please give the training and inference scripts a try, and feel free to leave an issue if you run into any problems. We apologize for any confusion caused by our original codebase release. `5.15, 2025`

🔥 We've released some demos. `5.22, 2025`

## 📺 Demo

🛋️ SOFA
![Demo1](https://github.com/Grady10086/DORAEMON/blob/master/demos/case1.gif)

🟦 TABLE
![Demo2](https://github.com/Grady10086/DORAEMON/blob/master/demos/case2.gif)

🛏️ BED
![Demo3](https://github.com/Grady10086/DORAEMON/blob/master/demos/case3.gif)

🌳 PLANT
![Demo4](https://github.com/Grady10086/DORAEMON/blob/master/demos/case4.gif)

🗄️ CABINET
![Demo5](https://github.com/Grady10086/DORAEMON/blob/master/demos/case5-min.gif)

💺 CHAIR
![Demo6](https://github.com/Grady10086/DORAEMON/blob/master/demos/case6.gif)

🌳 PLANT
![Demo7](https://github.com/Grady10086/DORAEMON/blob/master/demos/case7.gif)

🛋️ SOFA
![Demo8](https://github.com/Grady10086/DORAEMON/blob/master/demos/case8.gif)

📺 TV
![Demo9](https://github.com/Grady10086/DORAEMON/blob/master/demos/case9.gif)

🚽 TOILET
![Demo10](https://github.com/Grady10086/DORAEMON/blob/master/demos/case10.gif)

🛋️ SOFA
![Demo11](https://github.com/Grady10086/DORAEMON/blob/master/demos/case11.gif)

💺 CHAIR
![Demo12](https://github.com/Grady10086/DORAEMON/blob/master/demos/case12.gif)

## 🚀 Get Started

### ⚙️ Installation and Setup
1. Clone this repo.

2. Create the conda environment and install all dependencies.
    ```bash
    conda create -n doraemon python=3.9 cmake=3.14.0
    conda activate doraemon
    conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat
    pip install -r requirements.txt
    ```
   
### 🛢 Prepare Dataset
This project is based on [Habitat simulator](https://aihabitat.org/) and the HM3D and MP3D datasets are available [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).
Our code requires all above data to be in a data folder in the following format. Move the downloaded HM3D v0.1, HM3D v0.2 and MP3D folders into the following configuration:

```
├── <DATASET_ROOT>
│  ├── hm3d_v0.1/
│  │  ├── val/
│  │  │  ├── 00800-TEEsavR23oF/
│  │  │  │  ├── TEEsavR23oF.navmesh
│  │  │  │  ├── TEEsavR23oF.glb
│  │  ├── hm3d_annotated_basis.scene_dataset_config.json
│  ├── objectnav_hm3d_v0.1/
│  │  ├── val/
│  │  │  ├── content/
│  │  │  │  ├──4ok3usBNeis.json.gz
│  │  │  ├── val.json.gz
│  ├── hm3d_v0.2/
│  │  ├── val/
│  │  │  ├── 00800-TEEsavR23oF/
│  │  │  │  ├── TEEsavR23oF.basis.navmesh
│  │  │  │  ├── TEEsavR23oF.basis.glb
│  │  ├── hm3d_annotated_basis.scene_dataset_config.json
│  ├── objectnav_hm3d_v0.2/
│  │  ├── val/
│  │  │  ├── content/
│  │  │  │  ├──4ok3usBNeis.json.gz
│  │  │  ├── val.json.gz
│  ├── mp3d/
│  │  ├── 17DRP5sb8fy/
│  │  │  ├── 17DRP5sb8fy.glb
│  │  │  ├── 17DRP5sb8fy.house
│  │  │  ├── 17DRP5sb8fy.navmesh
│  │  │  ├── 17DRP5sb8fy_semantic.ply
│  │  ├── mmp3d.scene_dataset_config.json
│  ├── objectnav_mp3d/
│  │  ├── val/
│  │  │  ├── content/
│  │  │  │  ├──2azQ1b91cZZ.json.gz
│  │  │  ├── val.json.gz
```

### 🔑 Prepare Gemini API
You can set your own GeminiAPI key by `export GEMINI_API_KEY=xxx`

### 📈 Evaluation
Run `python scripts/main.py` to visualize the result of an episode.

To evaluate DORAEMON, we use a framework for parallel evaluation (HM3D v0.1 contains 1000 episodes, 2000 episodes for HM3D v0.2 and 2195 episodes for MP3D). The file `parallel_gpu0.sh` contains a script to distribute K instances over N GPUs, and for each of them to run M episodes. A local flask server is initialized to handle the data aggregation, and then the aggregated results are logged to wandb. Make sure you are logged in with `wandb login`.

## 📖 Citation

If you find our work useful, please cite:

```bibtex
@misc{gu2025doraemondecentralizedontologyawarereliable,
      title={DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation}, 
      author={Tianjun Gu and Linfeng Li and Xuhong Wang and Chenghua Gong and Jingyu Gong and Zhizhong Zhang and Yuan Xie and Lizhuang Ma and Xin Tan},
      year={2025},
      eprint={2505.21969},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.21969}, 
}
```

## 📫 Contact
For questions about this work, please contact:

Tianjun Gu: TianjunGu_Grady@outlook.com

Project Page: https://grady10086.github.io/DORAEMON/
<p align="center"> 
<img src="https://img.shields.io/github/stars/Grady10086/DORAEMON?style=social" alt="GitHub stars"> 
<img src="https://img.shields.io/github/forks/Grady10086/DORAEMON?style=social" alt="GitHub forks"> 
</p>