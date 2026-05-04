<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Privacy Leakage of iCloud Private Relay in the Era of Anonymous </h1>

<!-- <p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> [<a href="">Paper Link</a>] </p> -->

<!-- <p align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> <b>  </b> </p> -->

<!-- <p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> </p> -->


> [!NOTE]
> This is the **SupconNet attack model** proposed in *Privacy Leakage of iCloud Private Relay in the Era of Anonymous* work.


## 1. Environment
We utilized a workstation equipped with an Intel i5-13400F CPU, 64 GB of RAM, and an NVIDIA RTX 3080 GPU, running Ubuntu 22.04. The benchmarks were also run on Kaggle using GPU T4x2 and GPU P100.

## 2. Prerequisites and Settings

### 2-1. Python Dependencies

For experiments, we used the dependencies below:
```bash
tensorflow==2.6.0
keras==2.6.0
scikit-learn==1.3.0
numpy==1.22.4
pandas==2.2.2
tqdm==4.66.4
```

## 3. Dataset

For the datasets, you can use the download link below. Note that the datasets are in .pcap files.

| Dataset | Link | Size |
|-----|-----|-----|
| top-ranked keywords | [Link](https://drive.google.com/drive/folders/17ggEToGMdo1V0xwTjjltsVyJX-Mg0bkE?usp=sharing) | 50 classes * 500 instances |
| AOL search queries | [Link](https://drive.google.com/drive/folders/1g8rRtsD7hFOxJvjdRIIj67xxFhV6Q6XQ?usp=sharing) | 50 classes * 500 instances |


## 3. Run SupconNet and benchmarks

If you want to simply run the SupconNet model, use `SupconNet` folder.

If you want to run the others benchmarks we were mentioned in our research, use 'model' folder.


## 4. Contacts
Please contact us if you have any questions about SupconNet.

- Phuong-Linh Ha, linh.hp235598@sis.hust.edu.vn
- My Hoang Ha, my.hh235583@sis.hust.edu.vn
- Hong-Nhung Le, nhung.lh235393@sis.hust.edu.vn
- Van Tong, vantv@soict.hust.edu.vn
