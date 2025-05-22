#🔔DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation

##📚 Contents
- [Abstract](#Abstract)
- [Demo](#Demo)
- [Update](#Update)
- [Get Started](#Get-Started)

##Abstract
Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develop Nav-Ensurance to ensure navigation safety and efficiency.

##Demo
![Demo1]
![Demo2]

##Update
🔥We’ve reorganized and cleaned up the repository to ensure a clear, well-structured codebase. Please give the training and inference scripts a try, and feel free to leave an issue if you run into any problems. We apologize for any confusion caused by our original codebase release.


##Get-Started
###Installation and Setup
- clone this repo.

- Create the conda environment and install all dependencies.
conda create -n doraemon python=3.9 cmake=3.14.0
conda activate doraemon
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat

pip install -e .

pip install -r requirements.txt
