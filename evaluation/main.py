import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from matplotlib.pylab import cond
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from dataset.echodataset import EchoDataset
from evaluation.functions import load_model_from_run
from dataset.util import make_sampling_collate

run_dir = Path('outputs/hydra_outputs/2025-09-03/15-24-24')
model, cfg, ckpt_path = load_model_from_run(run_dir, ckpt_name=None)

test_ds = EchoDataset(cfg, split="test")
test_dl = DataLoader(test_ds, batch_size=1, collate_fn=make_sampling_collate(4, ef_gen_range=(0.1,0.9)))

device = next(model.parameters()).device
for reference_batch, repeated_batch in tqdm(test_dl, total=len(test_dl)):
    batch_size, *data_shape = repeated_batch['cond_image'].shape


    with torch.no_grad():
        sample_videos = model.sample(
            **repeated_batch,
            batch_size=batch_size,
            data_shape=data_shape
            )
    
    #TODO Decode videos


    #TODO Save them in the form 
    # dataset/
    #    video_x/
    #       frame_1.png
    #       frame_2.png
    #       ...
    #       frame_n.png

    a = 5

        


#TODO Take code from pl.sample to complete this section. Do reconstruction and generation in
# the same loop similar to sample example
#TODO for each model, for each flow type, for reconstruction and generation, for differing 
# number of unmasked frames
#TODO Some way of specifying n_missing_frames in 'mask_random_frames' so that test time can
# deterministic. t = 1 frame, 25%, 50%, 75%. So for each video you test, pass it through the
# model 4*4 times i.e. 16 different configurations.
