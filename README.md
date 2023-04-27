# Principled Adversarial Malware Detection
This code repository is for the paper, entitled **PAD:Towards Principled Adversarial Malware Detection** by Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Xiao Fu, Shouhuai Xu (Accepted by IEEE TDSC; To appear in future coming issue). Please check out the latest version of the paper on arxiv [here](https://arxiv.org/abs/2302.11328). 

## Overview
Our research motivation is to propose an adversarial training method for malware detection with provable and effective robustness. 
This means that we can religiously identify which kind of attacks the resultant model can defend against. 
The core innovation of our paper is that we propose a learnable convex measurement to quantify the distribution-wise discrete perturbations. 
Our contributions are summarized as follows:
* A new framework of adversarial training; 
* A new mixture of attacks for promoting the effectiveness of adversarial trining;
* Several groups of experiments for demonstrating the soundness of the new adversarial training method.

 The main features of this repository are listed as follows:
* Implement 7 defense models for Android malware detection;
* Implement 12 attack methods for Android malware detection.
* Implement an oracle by using Monkey Testing to justify the semantics-preservation of Android malware apps
* Generate the executable adversarial malware examples (APKs) automatically by using perturbations, including Manifest feature injection, API injection, API removal, and etc.
  
## Dependencies:
We develop codes on the system of **Ubuntu**. We recommend building a virtual conda environment by an instruction ```conda create -n adv_app python=3.6``` and then activating it by ```conda activate adv_app```,
where the ```adv_app``` is the name of the new conda environment.
The leveraged packages are as follows:
* torch>=1.7.1
* numpy >= 1.19.2
* scikit-Learn >= 0.23.2
* [androguard 3.3.5](https://github.com/androguard/androguard/releases/tag/v3.3.5)
* [apktool](https://ibotpeaches.github.io/Apktool/)

Most of dependencies can be installed by 'pip' (e.g., pip install -r requirements.txt), except for the toolkits of [apktool](https://ibotpeaches.github.io/Apktool/) and [pytorch](https://pytorch.org/get-started/locally/), both of which shall be installed by following the official document of its own.


## Usage
  #### 1. Dataset
  * We conduct the experiments upon the dataset [Drebin](https://www.sec.cs.tu-bs.de/~danarp/drebin/) and [Malcan](https://github.com/malscan-android/MalScan), respectively. 
    Both datasets are required to follow the policies of their own to obtain the apks. The sha256 checksums of benign apps in Drebin are available at [here](https://drive.google.com/drive/folders/1AHnNhtE2-YLWj8jeyciW52lFqFGdEmTB?usp=sharing). 
    These apks files can be downloaded from [Androzoo](https://androzoo.uni.lu/).
  * For reproducing the experimental results on the Drebin or Malscan dataset, we provide a portion of intermediate medium (e.g., vocabulary, dataset splitting info, etc.) available [here](https://drive.google.com/file/d/1JOiMzOjdgpyjM6WSmegpGmr6-32EEVYk/view?usp=share_link)
However, the data preprocessing step cannot be avoided, which means we need to download the apps and then conduct the experiments. We believe this is necessary because we attempt to generate the realistic attacks.
    
  #### 2. Configure
  We are required to change the `conf` by `project_root=/absolute/path/to/pad4amd/` and `database_dir = /absolute/path/to/datasets/` to accommodate the current project and dataset paths. To be special, in the folder of `database_dir`, the structure shall be:
  ```
  drebin
     |---attack.idx % APK names for waging attacks
     |---benign_samples % the folder contains benign apk files (optional if 'drebin' feature exists)
     |---malicious_samples % the folder contains malicious apk files (at least contains 800 APKs corresponding to the attack.list)
     |---attack % this folder contains attack results and will be created by default
  malscan
     |---attack.idx % APK names for waging attacks
     |---benign_samples % the folder contains benign apk files (optional if 'drebin' feature exists)
     |---malicious_samples % the folder contains malicious apk files (at least contains 800 APKs corresponding to the attack.list)
     |---attack % this folder contains attack results and will be created by default
  ```
 #### 3. Run some scripts
All scripts can be found in the folder ```pad4amd/``` with ```*.sh``` files
We suggest the following motions to perform the code: 

&emsp; (1). Learn defenses:

```
chmod +x run-drebin.sh; ./run-drebin.sh
``` 
&emsp; (2). Wage attacks against defenses:

&emsp; The first step is to change the model name in the script of ```run-attacks-drebin.sh``` corresponding to the learned models in the last step, e.g., changing 
```angular2html
python -m examples.mimicry_test --trials 10 --model "amd_pad_ma" --model_name "20220721-083300"
```
to 
```angular2html
python -m examples.mimicry_test --trials 10 --model "amd_pad_ma" --model_name "xxxxxxxx-xxxxxx"
```
wherein ```xxxxxxxx-xxxxxx``` should be changed. Then perform 
```
chmod +x run-attacks-drebin.sh; ./run-attacks-drebin.sh
```
&emsp; If the practical attacks are preferred, we shall append an extra command ```-r```  to the shell commands in the ```run-attacks-drebin.sh```, e.g.,
```angular2html
python -m examples.mimicry_test --trials 10 --model "amd_pad_ma" --model_name "20220721-083300" -r
```

## Learned Parameters

All learned model will be saved into the current directory under `save` folder that can be redirected by settings in the file of `conf`. We also provides some defenses models, which can be obtained for research (please contact me via e-mail: lideqiang@njupt.edu.cn).


## Contacts
If you have any questions or would like to make contributions to this repository such as [issuing](https://github.com/deqangss/pad4amd/issues) for us, please do not hesitate to contact us: `lideqiang@njupt.edu.cn`.

## License

* For ethical consideration, all the code presented on this repository is for educational/research proposes solely. The illegal or misuse of the code can lead to criminal behaviours. We (our organization and authors) will not be held responsible in any criminal charges.

* This project is released under the [GPL license](./LICENSE).

<!---
## Citation

If you'd like to cite us in a project or publication, please include a reference to the IEEE T-IFS paper (early access version):
```buildoutcfg
@ARTICLE{9121297,
  author={D. {Li} and Q. {Li}},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Adversarial Deep Ensemble: Evasion Attacks and Defenses for Malware Detection},
  year={2020},
  volume={15},
  number={},
  pages={3886-3900},
  doi={10.1109/TIFS.2020.3003571}
}
```
--->
