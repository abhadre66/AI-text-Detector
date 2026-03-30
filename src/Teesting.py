import pandas as pd

df = pd.read_csv("/Users/abhishekbhadre/Documents/Project/AI-text-Detector/data/raw/data2.csv")

#print(df.columns)
chatgpt_count = (df["label"] == 1).sum()
Human_count = (df["label"] == 0).sum()

print("AI (ChatGPT) samples:", chatgpt_count)
print("Human samples:", Human_count)