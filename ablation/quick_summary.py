import pandas as pd

df = pd.read_csv("ablation/results/ablation_raw.csv")
print(f"Total rows: {len(df)}, Images: {df['image'].nunique()}\n")

print(f"{'Model':<20} {'LLM':<10} {'Recall':>14}")
print("-" * 46)
for (m, l), g in df.groupby(["model", "llm"]):
    print(f"{m:<20} {l:<10} {g['flagged'].mean()*100:>13.1f}%")

if "inference_ms" in df.columns:
    print(f"\n{'Model':<20} {'Median ms':>10} {'Mean ms':>10}")
    print("-" * 42)
    for m, g in df[df["llm"] == "none"].groupby("model"):
        print(f"{m:<20} {g['inference_ms'].median():>9.1f}ms {g['inference_ms'].mean():>9.1f}ms")
