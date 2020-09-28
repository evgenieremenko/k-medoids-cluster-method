from scipy.spatial.distance import cdist
import numpy as np

size = 103
a = np.random.random((size,size)).astype(np.float32)
out = np.empty((size,size), np.float32).astype(np.float32)

width = 16
iterations = size//width if size%width==0 else size//width+1
for i in range(iterations):
    for j in range(iterations):
        sub_a = a[width*i:width*i+width]
        sub_b = a[width*j:width*j+width]
        temp = cdist(sub_a,sub_b, "euclidean")
        print(f"putting it in big matrix....temp_shape={temp.shape} and sub_a = {sub_a.shape}, sub_b={sub_b.shape}")
        print("temp.......\n",temp)
        print("out before.....\n",out[width*i:width*i+width,width*j:width*j+width])
        out[width*i:width*i+width,width*j:width*j+width] = temp
        print("out after.....\n",out[width*i:width*i+width,width*j:width*j+width])
        print('\n\n')


    # print(f"small_a: {small_a.shape} and small_b: {small_b.shape}", end = "   ")
    # print("#"*10, f"row_start={width*i} and row_end={width*i+width}  and col_start={width*j} and col_end={width*j+width}")
