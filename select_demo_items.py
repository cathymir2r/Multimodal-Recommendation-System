import pandas as pd
import re

DATA_CSV_PATH = "amazon_clothing.csv"
NUM_DEMO_ITEMS = 12

# 一些常见“看起来就是服装/配饰”的关键词
CLOTHING_KEYWORDS = [
    r"dress", r"skirt", r"tutu", r"shirt", r"t[- ]?shirt",
    r"jeans", r"pants", r"shorts", r"coat", r"jacket",
    r"hoodie", r"sweater", r"cardigan", r"blouse",
    r"shoes?", r"boot[s]?", r"sandals?", r"sneakers?",
    r"nightgown", r"pajama", r"sleepwear", r"robe",
    r"swim", r"bikini", r"legging", r"sock[s]?",
    r"hat", r"cap", r"scarf", r"glove[s]?", r"belt"
]
pattern = re.compile("|".join(CLOTHING_KEYWORDS), flags=re.IGNORECASE)


def main():
    df = pd.read_csv(DATA_CSV_PATH)

    # 只保留 item 级别信息
    items = df[["item_id", "title"]].drop_duplicates(subset=["item_id"]).copy()

    # 用标题关键词粗略筛掉“软件/珠宝盒”之类非服装
    mask = items["title"].astype(str).apply(lambda x: bool(pattern.search(x)))
    clothing_items = items[mask].copy()

    print(f"总 item 数: {len(items)}, 服装类候选: {len(clothing_items)}")

    # 取前 NUM_DEMO_ITEMS 个（也可以 .sample(random_state=42) 随机抽样）
    demo_items = clothing_items.head(NUM_DEMO_ITEMS).reset_index(drop=True)

    print("\n推荐用于展示的 demo item 列表：")
    for i, row in demo_items.iterrows():
        print(f"{i+1:2d}. item_id = {row['item_id']}, title = {row['title']}")

    # 也写到文件，方便后面查
    demo_items.to_csv("demo_items_for_images.csv", index=False)
    print("\n已将结果保存到 demo_items_for_images.csv")


if __name__ == "__main__":
    main()