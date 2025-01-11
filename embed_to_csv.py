import sys

import numpy as np
import pandas as pd

filename = "frames_image_embeddings.npy" if len(sys.argv) < 2 else sys.argv[1]
out = "out.csv" if len(sys.argv) < 3 else sys.argv[2]

df = pd.DataFrame(np.load(filename))
df["frame"] = [f"frame_{i:06d}" for i in range(df.shape[0])]
df = df.set_index("frame")
print(df)
df.to_csv(out)
