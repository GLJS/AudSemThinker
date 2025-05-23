import polars as pl

data = pl.read_csv("classification/bert-base-uncased-output.csv")

print("data_shape:", data.shape)

sounds = data.filter(pl.col("predicted_class") == 1)
print("sounds_shape:", sounds.shape)

sounds.write_csv("classification/bert-sounds.csv")