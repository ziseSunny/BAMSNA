# BAMSNA

Run the code for BAMSNA (currently support the four datasets mentioned in paper):

```
python main.py --dataset dblp/douban/AI/wd [--alpha 0.02 --beta 0.02 --known_ratio 0.1...]
```

We only store ACM-DBLP (ACM-DBLP_0.2.npz), Online-Offline (Douban.mat), and Weibo-Douban (MAUIL_douban_weibo.npz) in "/data", since Allmovie-IMDB (Allmovie_Imdb.npz of over 500MB) exceeds the uploading limit (100MB). In specific, Allmovie_Imdb.npz is extracted from the original one in GAlign: https://github.com/vinhsuhi/GAlign.

Machine configuration:

```
python==3.6.13, torch==1.8.0, torch-geometric==1.4.3, NVIDIA Tesla T4
```