YOLOV5_GUIDE项目 说明文档
===========================
该文档用于说明yolov5_guide的使用介绍，以便聚均科技同事将来涉及Yolov5的工作。

****
	
| 项目 | YOLOV5_GUIDE |
| ---- | ---- |
| 项目版本 | v3.0 |
| YOLOv5版本 | Jun 20, 2021 |
| 完成时间 | 2021-02-12 |


****
# 目录
* [一、训练检测部分：快速上手入门](#一、训练检测部分：快速上手入门)
* [二、数据集部分：数据集结构、拆分、打标与建yaml](#二、数据集部分：数据集结构、拆分、打标与建yaml)
    * 1.数据集结构
    * 2.拆分方式
    * 3.打标方式
    * 4.建yaml
* [三、参数部分：新手需了解的部分参数](#三、参数部分：新手需了解的部分参数)
    * 1.train
    * 2.detect
* [四、YOLOv5文档](#四、YOLOv5文档)
* [五、参考文献](#五、参考文献)

****


# 一、训练检测部分：快速上手入门

> **Step 1: 先从windows环境登录到服务器，可使用各种类shell工具或ssh软件，以最基础的cmd举例如下，账户密码需要联系汤平开通**  
```bash
$ ssh xuenze@10.30.4.96
```

> **Step 2: 新建工作目录，例如test_workspace，然后进入工作目录**  
```bash
xuenze@bigdata_gpu:~$ mkdir test_workspace
xuenze@bigdata_gpu:~$ cd test_workspace
```

> **Step 3: 将我们的示例代码clone到本地（如已clone，可以跳过），然后进入目录**  
```bash
xuenze@bigdata_gpu:~/test_workspace$ git clone http://git.fusionfintrade.com/xuenze/yolov5_guide.git
xuenze@bigdata_gpu:~/test_workspace$ cd yolov5_guide/yolov5
```

> **Step 4: 进入yolov5的已经配置好的虚拟环境**  
```bash
xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ source /home/zhuyouzhi/.bashrc
(base) xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ source activate yolov5
```

> **Step 5: 进行训练，训练结束后模型会出现在runs/train/exp/weights/best.pt，500images+50个epochs的gpu训练时间约为2-3min（多次训练后为exp2, exp3, ...自动递增）**  
```bash
(yolov5) xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ python3 train.py --data demo_data.yaml --epochs 50 --weights yolov5s.pt --device 0
```

> **Step 6: 进行检测，使用我们刚才训练好的模型，检测使用数据../demo_test中的20幅图片，结果自动存放在runs/detect/exp/中（多次检测后为exp2, exp3, ...自动递增）**  
```bash
(yolov5) xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ python3 detect.py --source ../demo_test --weights runs/train/exp/weights/best.pt --device 0
```

> **Step 7: 退出虚拟环境**  
```bash
(yolov5) xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ conda deactivate
(base) xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ conda deactivate
xuenze@bigdata_gpu:~/test_workspace/yolov5_guide/yolov5$ cd ~
```

# 二、数据集部分：数据集结构、拆分、打标与建yaml

## 1. 数据集结构
> **结构必须严格按照以下结构编排，示例可参见一中使用的../demo_test。使用本章第3节中的labelImg会自动生成图片对应的labels内容** 
```text
\-- test123
    \-- images
        \-- train
            \-- t1.jpg
            \-- t2.jpg
            \-- ...
            \-- t1000.jpg
        \-- valid
            \-- v1.jpg
            \-- v2.jpg
            \-- ...
            \-- v1000.jpg
    \-- labels
        \-- train
            \-- classes.txt
            \-- t1.txt
            \-- t2.txt
            \-- ...
            \-- t1000.txt
        \-- valid
            \-- classes.txt
            \-- v1.txt
            \-- v2.txt
            \-- ...
            \-- v1000.txt
```
## 2. 拆分方式
> **当数据量比较小时，可以使用7:3(train:validation)，或者6:2:2(train:validation:test)训练数据。当数据量非常大时，可以使用98:1:1**


## 3. 打标方式
> **先下载labelImg，命令行可以直接启动，打开目录，矩形框选中范围，目标文件夹会自动生成对应的.txt和classes.txt**
```bash
$ pip install labelImg -i https://mirrors.aliyun.com/pypi/simple/
$ labelImg
```

## 4. 建yaml
> **标注完数据并建立数据集结构之后，最后一步是建yaml文件，用于让YOLOv5"看懂"你的数据集结构与目标。这里我们以demo_data对应的data/demo_data.yaml为例**
```yaml
# 用于指定train和validation的主路径
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../demo_data/images/train/  # 300 images
val: ../demo_data/images/valid/  # 150 images

# 用于指定目标种类数
# number of classes
nc: 1

# 用于指定每个目标种类的名字，注意要与打标中设定的classes.txt中的目标名字相同，多项需要以['aa', 'bbb', 'cccc']这样的python格式编写
# class names
names: ['paper']
```

# 三、参数部分：新手需了解的部分参数
> **简单介绍一些常用参数的意义，更多完整参数请参照第4章**

## 1. train
### 1.1 train部分常用参数
```bash
$ python train.py --img 640 --batch 16 --epochs 5 --data test123.yaml --weights yolov5s.pt --device 0
# --img: 设置的图片大小为640*640
# --batch: batch size（如果内存不足则将其设小）
# --epochs: epochs次数设定，太小训练量不足，太大可能发生过拟合现象，个人建议1000以内规模数据集选300-500之间
# --data: 使用的数据集的yaml文件
# --weights: 使用的预训练权重文件，有yolov5s.pt, yolov5x.pt等，yolov5s.pt速度最快
# --device: 使用设备，cpu或者0, 1, 2,...（我们服务器中0为gpu）
```
### 1.2 train部分Parse源码
```python
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```

## 2. detect
### 2.1 detect部分常用参数
```bash
$ python detect.py --source test123 --weights ./yolov5s.pt --device 0
# --source: 待检测的图片/目录路径，若为目录则逐一检测其中每一幅图片
# --weights: 使用的预训练权重文件，有yolov5s.pt, yolov5x.pt等，yolov5s.pt速度最快
# --device: 使用设备，cpu或者0, 1, 2,...（我们服务器中0为gpu）
```
### 2.2 detect部分Parse源码
```python
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt
```


# 四、YOLOv5文档
<div align="center">
<p>
<a align="left" href="https://ultralytics.com/yolov5" target="_blank">
<img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>

<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<!-- 
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>


## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.


## <div align="center">Quick Start Examples</div>


<details open>
<summary>Install</summary>

Python >= 3.6.0 required with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed:
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```
</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36). Models automatically download from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, PIL, OpenCV, numpy, multiple

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>  

<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️ RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

</details>


## <div align="center">Environments and Integrations</div>

Get started in seconds with our verified environments and integrations, including [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme) for automatic YOLOv5 experiment logging. Click each icon below for details.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-small.png" width="15%"/>
    </a>
</div>  


## <div align="center">Compete and Win</div>

We are super excited about our first-ever Ultralytics YOLOv5 🚀 EXPORT Competition with **$10,000** in cash prizes!  

<div align="center">
<a href="https://github.com/ultralytics/yolov5/discussions/3213">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"/>
</a>
</div>


## <div align="center">Why YOLOv5</div>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>
  
<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313219-f1d70e00-9af5-11eb-9973-52b1f98d321a.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>
  
  * GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. 
  * EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.
  * **Reproduce** by `python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`
</details>

### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                    |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s][assets]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m][assets]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l][assets]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x][assets]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6][assets]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6][assets]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6][assets]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6][assets]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-

<details>
  <summary>Table Notes (click to expand)</summary>
  
  * AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
  * AP values are for single-model single-scale unless otherwise noted. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
  * Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes FP16 inference, postprocessing and NMS. **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
  * All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
  * Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) includes reflection and scale augmentation. **Reproduce TTA** by `python test.py --data coco.yaml --img 1536 --iou 0.7 --augment`
</details>


## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see our [Contributing Guide](CONTRIBUTING.md) to get started. 


## <div align="center">Contact</div>

For issues running YOLOv5 please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business or professional support requests please visit 
[https://ultralytics.com/contact](https://ultralytics.com/contact).


# 五、参考文献
[1. 本项目gitlab链接：http://git.fusionfintrade.com/xuenze/yolov5_guide](http://git.fusionfintrade.com/xuenze/yolov5_guide)

[2. YOLOv5原项目链接: https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

[3. 其它供参考的YOLOv5教程: https://www.bilibili.com/read/cv8142756/](https://www.bilibili.com/read/cv8142756/)


