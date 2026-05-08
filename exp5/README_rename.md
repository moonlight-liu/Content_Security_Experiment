# 批量重命名说明

放在 `exp5/rename_face_db.py` 的脚本用于把 `exp5/face_db` 下形如 `202330218107-刘子明-网络空间安全-男.jpg` 的文件重命名为

```
学号_英文名_Cyberspace Security_性别.jpg
```

使用示例：

```bash
# 进入项目根目录
python exp5/rename_face_db.py --dir exp5/face_db --map exp5/face_db/name_map.csv

# 先做一次演练（不实际重命名）
python exp5/rename_face_db.py --dir exp5/face_db --map exp5/face_db/name_map.csv --dry
```

说明：
- 若 `name_map.csv` 中存在中文名到英文名的映射，会优先使用该映射（格式：cn_name,en_name）。
- 若不存在，会尝试使用 `pypinyin` 将中文名拼音化（建议安装：`pip install pypinyin`），若未安装会使用简单替代方案。
- 脚本只做重命名（包括把扩展名改为 `.jpg`），不做图像格式转换。
