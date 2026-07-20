# 照片整理工具 (Photo Organizer)

本地 AI (CLIP) 照片整理：人像/风景分类 + 相似风景聚类 + 网页浏览管理。照片不上传、全程离线。

## 文件说明

| 文件 | 用途 |
|---|---|
| `organize_photos.py` | 主整理脚本：扫描 → CLIP 编码 → 人像/风景分类 → 风景聚类 → 复制落地 |
| `label_clusters.py` | 给 `cluster_XXX` 文件夹自动起中文名（如 `cluster_003_雪山`），需插上照片硬盘 |
| `app.py` + `static/` | 本地网页应用：浏览缩略图、移动照片、重命名文件夹 |
| `start_photo_app.bat` | 双击启动网页应用（仅本机访问） |
| `start_photo_app_lan.bat` | 双击启动网页应用（手机同 WiFi 可访问，首次运行允许防火墙） |
| `label_clusters.bat` | 双击运行聚类自动命名 |

## 日常用法

1. 插上照片硬盘 (D:)
2. 双击 `start_photo_app.bat`，浏览器自动打开 http://127.0.0.1:8765
3. 手机访问：改用 `start_photo_app_lan.bat`，手机浏览器打开窗口里打印的 `http://192.168.x.x:8765`

网页功能：
- 顶部两个下拉框选 地区 → 文件夹，缩略图网格浏览，点图看大图（左右点切换）
- 「选择」→ 勾选照片 →「移动到…」把分错的照片挪到正确文件夹（配对 RAW/XMP 自动跟随）
- 「重命名文件夹」给当前聚类文件夹起名

## 新照片再次整理

```powershell
cd C:\Users\ghs\photo-organizer
.\.venv\Scripts\python.exe organize_photos.py --input "D:\新照片文件夹" --output "D:\新照片_整理" --region-mode
```

常用参数：`--min-cluster-size 8`（聚类更粗）、`--move`（移动而非复制）、`--limit 100`（试跑）。
已算过的照片有缓存（`.embeddings_cache.pkl`），重跑聚类秒级完成。

## 输出结构

```
输出根目录\
  <地区>\people\              人像
  <地区>\uncertain\           AI 不确定（人工复核）
  <地区>\scenery\cluster_XXX\ 相似风景一组
  <地区>\scenery\misc\        零散风景
  manifest.json               每张图 原路径→去向 记录
```
