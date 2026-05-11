import os
import json
import gzip
from typing import List, Dict

import pandas as pd

# 原始 JSON 文件路径（根据你的实际路径修改）
RAW_JSON_PATH = r"D:\多模态推荐系统\Clothing_Shoes_and_Jewelry_5.json"
# 如果你下载的是 .json.gz，就改成下面这样：
# RAW_JSON_PATH = r"D:\多模态推荐系统\Clothing_Shoes_and_Jewelry_5.json.gz"

# 输出的 CSV 文件名（main.py 默认会读这个）
OUTPUT_CSV_PATH = r"D:\多模态推荐系统\amazon_clothing.csv"

# 为了便于调试，可以先只取前 N 条；如果想全量就设为 None
MAX_ROWS = 50000  # 或者 None


def read_json_lines(path: str, max_rows=None) -> List[Dict]:
    """读取 Amazon JSON（每行一个 JSON 对象）的文件"""
    data = []
    open_fn = gzip.open if path.endswith(".gz") else open

    with open_fn(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                # 个别坏行直接跳过
                continue
    return data


def main():
    if not os.path.exists(RAW_JSON_PATH):
        raise FileNotFoundError(f"找不到原始 JSON 文件: {RAW_JSON_PATH}")

    print(f"读取原始 JSON 数据: {RAW_JSON_PATH}")
    records = read_json_lines(RAW_JSON_PATH, max_rows=MAX_ROWS)
    print(f"共读取记录数: {len(records)}")

    # 原始字段示例（UCSD 数据）：reviewerID, asin, overall, reviewText, summary, unixReviewTime, reviewTime, ...
    # 我们映射成自己需要的字段：
    # user_id, item_id, title, description, review_text, image_path, brand, price, category, rating, time
    rows = []
    for obj in records:
        user_id = obj.get("reviewerID", "")
        item_id = obj.get("asin", "")
        rating = obj.get("overall", None)
        review_text = obj.get("reviewText", "")
        summary = obj.get("summary", "")
        unix_time = obj.get("unixReviewTime", None)

        # 这些字段原始数据里可能没有，这里先留空（后续如果你整合 metadata，再补充也行）
        description = ""
        image_path = ""   # 暂时没有图片路径，后面有需要可以自己补
        brand = ""
        price = None
        category = "Clothing_Shoes_and_Jewelry"

        row = {
            "user_id": user_id,
            "item_id": item_id,
            "title": summary,
            "description": description,
            "review_text": review_text,
            "image_path": image_path,
            "brand": brand,
            "price": price,
            "category": category,
            "rating": rating,
            "time": unix_time,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("前 5 行数据预览：")
    print(df.head())

    # 保存成 CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n已保存到 CSV: {OUTPUT_CSV_PATH}")
    print(f"总行数: {len(df)}")


if __name__ == "__main__":
    main()