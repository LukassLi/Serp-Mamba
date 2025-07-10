Environment Installation
Requirements: Ubuntu 20.04, CUDA 12.2
	1. Create a virtual environment: conda create -n Serp-mamba python=3.10 -y and conda activate Serp-mamba, cd SerpMamba
	2. Pytorch : pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
	3. Install mamba_ssm and causal-conv1d: download causal_conv1d-1.1.3.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
	and mamba_ssm-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl. Then insert pip install causal_conv1d-1.1.3.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86	_64.whl and pip install mamba_ssm-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
	4. pip install -r requirements.txt
Train your Serp-Mamba
	python train_single_rater.py
Test your Serp-Mamba model
	python test_single_rater.py
	