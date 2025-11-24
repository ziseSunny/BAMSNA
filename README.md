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


User indexes and number of ground-truth for ACM-DBLP, Online-Offline, Allmovie-IMDB, and Weibo-Douban:

```
ACM-DBLP_0.2.npz: ACM (0~9871, 9872 users in total), DBLP (0~9915, 9916 users in total), ground-truth (1265 pos_pairs + 5060 neg_pairs = 6325 pairs in total)
Douban.mat: Online (0~3905, 3906 users in total), Offline (0~1117, 1118 users in total), ground-truth (1118 pairs in total)
Allmovie-Imdb.npz: Allmovie (0~6010, 6011 users in total), IMDB (0~5712, 5713 users in total), ground-truth (5176 pairs in total)
MAUIL_douban_weibo.npz: Weibo (0~9713, 9714 users in total), Douban (0~9525, 9526 users in total), ground-truth (1397 pairs in total)
```


Allmovie-Imdb.npz can be obtained via: https://pan.baidu.com/s/17A_CLCmG6GBgpEHzMhT5og by code: kuha.