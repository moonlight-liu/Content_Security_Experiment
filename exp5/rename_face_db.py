#!/usr/bin/env python3
"""
批量重命名 exp5/face_db 下的图片文件。

原始文件名格式示例:
  202330218107-刘子明-网络空间安全-男.jpg

目标格式:
  学号_英文名_Cyberspace Security_性别.jpg

规则:
- 从文件名中解析学号、中文名、专业、性别
- 专业统一写成 "Cyberspace Security"
- 性别: "男"->"male", "女"->"female"，其它保持原文
- 中文名先在映射文件 `name_map.csv` 中查找（格式: cn_name,en_name），如果缺失，尝试使用 `pypinyin` 转写
- 新文件扩展名统一为 `.jpg`（只是重命名，不做格式转换）

用法:
  python rename_face_db.py --dir exp5/face_db --map exp5/face_db/name_map.csv

"""

import os
import re
import csv
import argparse
import sys

try:
    from pypinyin import pinyin, Style

    HAS_PYPINYIN = True
except Exception:
    HAS_PYPINYIN = False


def load_name_map(map_path):
    d = {}
    if not map_path or not os.path.isfile(map_path):
        return d
    try:
        with open(map_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2:
                    cn = row[0].strip()
                    en = row[1].strip()
                    if cn:
                        d[cn] = en
    except Exception as e:
        print(f"Warning: failed to read name map {map_path}: {e}")
    return d


def chinese_to_pinyin(name):
    if not name:
        return ""
    if HAS_PYPINYIN:
        parts = pinyin(name, style=Style.NORMAL)
        flattened = [p[0] for p in parts]
        # Capitalize each part and join with underscore
        return "_".join([w.capitalize() for w in flattened])
    else:
        # Fallback: replace non-ascii with underscores (best-effort)
        safe = []
        for ch in name:
            if ch.isascii() and ch.isalnum():
                safe.append(ch)
            else:
                safe.append("_")
        s = "".join(safe)
        s = re.sub("_+", "_", s).strip("_")
        return s or name


def gender_map(g):
    if not g:
        return g
    g = g.strip()
    if g in ("男", "M", "m", "Male", "male"):
        return "male"
    if g in ("女", "F", "f", "Female", "female"):
        return "female"
    return g


def safe_filename(name):
    # remove characters that are problematic in filenames
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[:*?"<>|]', "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def make_new_name(student_id, cn_name, name_map, gender):
    # get english name
    en_name = name_map.get(cn_name)
    if not en_name:
        en_name = chinese_to_pinyin(cn_name)
    en_name = safe_filename(en_name)
    major = "Cyberspace Security"
    g = gender_map(gender)
    new = f"{student_id}_{en_name}_{major}_{g}.jpg"
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default="exp5/face_db", help="folder containing images"
    )
    parser.add_argument(
        "--map",
        default="exp5/face_db/name_map.csv",
        help="optional CSV map cn_name,en_name",
    )
    parser.add_argument("--dry", action="store_true", help="dry run (do not rename)")
    args = parser.parse_args()

    folder = args.dir
    if not os.path.isdir(folder):
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    name_map = load_name_map(args.map)

    # regex: id-cnname-major-gender.ext
    pattern = re.compile(
        r"^(?P<id>\d+)-(?P<cn>[^-]+)-(?P<major>[^-]+)-(?P<gender>[^.]+)\.(?P<ext>.+)$"
    )

    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        if fname == os.path.basename(args.map):
            continue
        m = pattern.match(fname)
        if not m:
            print(f"Skipping (pattern not matched): {fname}")
            continue

        sid = m.group("id")
        cn = m.group("cn")
        gender = m.group("gender")

        newname = make_new_name(sid, cn, name_map, gender)
        newpath = os.path.join(folder, newname)

        # avoid overwrite
        base, ext = os.path.splitext(newpath)
        i = 1
        while os.path.exists(newpath):
            newpath = f"{base}_{i}{ext}"
            i += 1

        print(f"Renaming: {fname} -> {os.path.basename(newpath)}")
        if not args.dry:
            try:
                os.rename(fpath, newpath)
            except Exception as e:
                print(f"Failed to rename {fname}: {e}")


if __name__ == "__main__":
    main()
