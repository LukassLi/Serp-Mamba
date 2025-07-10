# Serp-Mamba
Serp-Mamba: Advancing High-Resolution Retinal Vessel Segmentation with Selective State-Space Model

# How to Run the Code 🛠
## Environment Installation
Requirements: Ubuntu 20.04, CUDA 12.2

	1. Create a virtual environment: conda create -n Serp-mamba python=3.10 -y and conda activate Serp-mamba, cd SerpMamba
 
	2. Pytorch : pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
 
	3. Install mamba_ssm and causal-conv1d: download causal_conv1d-1.1.3.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
	and mamba_ssm-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl. Then insert pip install causal_conv1d-1.1.3.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86	_64.whl and pip install mamba_ssm-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
 
	4. pip install -r requirements.txt

 ### 1. Train Serp-Mamba
	python train.py

 ### 2. Test Serp-Mamba
	python test.py
	

# Dataset 📊
Multi-center UWF-SLO Vessel Segmentation (MU-VS) dataset 
<img width="1396" alt="data" src="https://github.com/user-attachments/assets/585243ea-4da6-403a-b831-9b504af9ae1f">

For PRIME-FP20: please refer to this [IEEE article](https://ieeexplore.ieee.org/abstract/document/9208757) and [dataset link](https://ieee-dataport.org/open-access/prime-fp20-ultra-widefield-fundus-photography-vessel-segmentation-dataset).

For Center A and Center B:
please contact Hongqiu (hongqiuwang16@gmail.com) for the dataset. One step is needed to download the dataset: **1) Use your google email to apply for the download permission ([OneDrive](https://hkustgz-my.sharepoint.com/my?id=%2Fpersonal%2Fhwang007%5Fconnect%5Fhkust%2Dgz%5Fedu%5Fcn%2FDocuments%2F24MICCAI%2DSFADA%2DUWF&sortField=FileLeafRef&isAscending=true), [BaiduPan-A](https://pan.baidu.com/s/1ESNR_tAFWhK6tfWzcMZuHg), [BaiduPan-B](https://pan.baidu.com/s/1v_Xr48SSxgvAIbT4t_aHMw)). We just handle the **real-name email** and **your email suffix must match your affiliation**. The email should contain the following information:


    Name/Homepage/Google Scholar: (Tell us who you are.)
    Primary Affiliation: (The name of your institution or university, etc.)
    Job Title: (E.g., Professor, Associate Professor, Ph.D., etc.)
    Affiliation Email: (the password will be sent to this email, we just reply to the email which is the end of "edu".)
    How to use: (Only for academic research, not for commercial use or second-development.)
**The data provided cannot be forwarded to others, and only individuals with approved applications are authorized to use them.**

**Thanks for understanding and cooperation!**

# Citation 📖

If you find our work useful or relevant to your research, please consider citing:
```
@article{wang2024serp,
  title={Serp-Mamba: Advancing High-Resolution Retinal Vessel Segmentation with Selective State-Space Model},
  author={Wang, Hongqiu and Chen, Yixian and Chen, Wu and Xu, Huihui and Zhao, Haoyu and Sheng, Bin and Fu, Huazhu and Yang, Guang and Zhu, Lei},
  journal={arXiv preprint arXiv:2409.04356},
  year={2024}
}
```
We thank the authors of nnU-Net, Mamba, U-mamba, and DSCNet for making their valuable code.
