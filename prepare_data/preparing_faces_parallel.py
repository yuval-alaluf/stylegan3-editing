import math
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path

import pyrallis

import dlib
from dataclasses import dataclass

sys.path.append(".")
sys.path.append("..")

from configs.paths_config import model_paths
from utils.alignment_utils import align_face, crop_face

SHAPE_PREDICTOR_PATH = model_paths["shape_predictor"]


@dataclass
class Options:
    # Number of threads to run in parallel
    num_threads: int = 1
    # Path to raw data
    root_path: str = ""
    # Should be 'align' / 'crop'
    mode: str = "align"
    # In case of cropping, amount of random shifting to perform
    random_shift: float = 0.05


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_on_paths(file_paths, args: Options):

    predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
    detector = dlib.get_frontal_face_detector()
    pid = mp.current_process().name
    print(f'\t{pid} is starting to extract on #{len(file_paths)} images')
    tot_count = len(file_paths)
    count = 0
    for file_path, res_path in file_paths:
        count += 1
        if count % 100 == 0:
            print(f'{pid} done with {count}/{tot_count}')
        try:
            if args.mode == "align":
                res = align_face(file_path, detector, predictor)
            else:
                res = crop_face(file_path, detector, predictor, random_shift=args.random_shift)
            res = res.convert('RGB')
            Path(res_path).parent.mkdir(exist_ok=True, parents=True)
            res.save(res_path)
        except Exception:
            continue
    print('\tDone!')


@pyrallis.wrap()
def run(args: Options):

    assert args.mode in ["align", "crop"], "Expected extractions mode to be one of 'align' or 'crop'"

    root_path = Path(args.root_path)
    out_crops_path = root_path.parent / Path(root_path.name + "_" + args.mode + "ed")
    if not out_crops_path.exists():
        out_crops_path.mkdir(exist_ok=True, parents=True)

    file_paths = []
    for file in root_path.iterdir():
        res_path = out_crops_path / file.name
        file_paths.append((str(file), str(res_path)))

    file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
    print(len(file_chunks))
    pool = mp.Pool(args.num_threads)
    print(f'Running on {len(file_paths)} paths\nHere we goooo')
    tic = time.time()
    pool.map(partial(extract_on_paths, args=args), file_chunks)
    toc = time.time()
    print(f'Mischief managed in {tic - toc}s')


if __name__ == '__main__':
    run()
