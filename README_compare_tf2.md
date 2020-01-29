Installing the environment (numpy need python 3.5):
```
conda create -n python35_clean python=3.5
pip install matplotlib tensorflow-gpu==1.13.1 chainer==3.3.0 numpy==1.11.1 cython==0.27.2 cupy==2.0.0 scipy==0.19.0 pillow==4.3.0 pyyaml==3.12 h5py==2.7.1 
pip install pandas
```

Download inception V3 model:
```
cd ./source/inception
python download.py --outfile ./../../datasets/inception_model
```


Run unsupervised cifar10
```
python train.py --gpu 0 --config_path ./configs/sn_cifar10_unconditional.yml --results_dir ./results/unsup1
```

Run metric:
```
python inception_score_comapre.py --gpu 0 --results_dir ./results/unsup1
```
