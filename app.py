from fastapi import FastAPI
import uvicorn
from typing import Union
from joblib import dump, load
import faiss
import numpy as np

app = FastAPI()
dims = 72
faiss_index = None
scaler_SS = None
mapper = {}


def parse_string(vec: str) -> list[float]:
    l = vec.split(",")
    if len(l) != dims:
        return None
    return [float(el) for el in l]


@app.on_event("startup")
def start():
    global faiss_index
	global scaler_SS
    n_cells = 5
    scaler_filename = '/content/drive/MyDrive/std_scale_1.bin'
	index_filename = '/content/drive/MyDrive/idx_l2_200_20.index'
	
    try:
	    idx_l2 = faiss.read_index(index_filename)
		scaler_SS=load(scaler_filename)
	except:
        return {"status": "fail", "message": "No index data"}


@app.get("/")
def main() -> dict:
    return {"status": "OK", "message": "Hello, world!"}


@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    global faiss_index
    global scaler_SS	
	
    if item is None:
        return {"status": "fail", "message": "No input data"}

    vec = parse_string(item)
    vec = np.ascontiguousarray(vec, dtype="float")[np.newaxis, :]
    # df_validation_SS = scaler_SS.transform(df_validation)

	r, idx = idx_l2.search(vec, 5)

    return {"status": "OK", "data": [str(el) for el in idx]}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=1030)