import pandas as pd
# Moyenne BLOSUM par type dans ton CSV
df = pd.read_csv("../results/tp53_mutations_critiques.csv")
f=df.groupby("impact")["blosum"].mean()
print(f)