
'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
import csv
import json
from soundfile import SoundFile
from pesq import pesq
from pystoi import stoi
from concurrent.futures import ProcessPoolExecutor

from mbstoi.mbstoi import mbstoi

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_audio(filename):
    """Read a wavefile and return as numpy array of floats.
            Args:
                filename (string): Name of file to read
            Returns:
                ndarray: audio signal
            """
    try:
        wave_file = SoundFile(filename)
    except:
        # Ensure incorrect error (24 bit) is not generated
        raise Exception(f"Unable to read {filename}.")
    return wave_file.read()

def compute_pesq(target, enhanced, sr, mode):
    """Compute PESQ from: https://github.com/ludlows/python-pesq/blob/master/README.md
        Args:
            target (string): Name of file to read
            enhanced (string): Name of file to read
            sr (int): sample rate of files
            mode (string): 'wb' = wide-band (16KHz); 'nb' narrow-band (8KHz)
        Returns:
            PESQ metric (float)
                """
    return pesq(sr, target, enhanced, mode)

def compute_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
           Args:
               target (string): Name of file to read
               enhanced (string): Name of file to read
               sr (int): sample rate of files
           Returns:
               STOI metric (float)
                   """
    return stoi(target, enhanced, sr)

def compute_mbstoi(clean_signal, enhanced_signal, sr):
    """compute MBSTOI"""
    left_ear_clean = clean_signal[:,0]
    right_ear_clean = clean_signal[:,1]
    left_ear_noisy= enhanced_signal[:,0]
    right_ear_noisy= enhanced_signal[:,1]

    #to modify mbstoi parameters see mbstoi/parameters.yaml
    mbstoi_score = mbstoi(left_ear_clean, right_ear_clean, left_ear_noisy, right_ear_noisy, sr_signal=sr)  # signal sample rate
    return mbstoi_score


def run_metrics(scene, enhanced, target, cfg):

    # Retrieve the scene name
    scene_name = scene["scene"]

    enh_file = os.path.join(enhanced, f"{scene_name}{cfg['enhanced_suffix']}.wav")
    tgt_file = os.path.join(target, f"{scene_name}{cfg['target_suffix']}.wav")
    scene_metrics_file = os.path.join(cfg["metrics_results"], f"{scene_name}.csv")

    # Skip processing with files don't exist or metrics have already been computed
    if ( not os.path.isfile(enh_file) ) or ( not os.path.isfile(tgt_file) ) or ( os.path.isfile(scene_metrics_file)) :
        return

    # Read enhanced signal
    enh = read_audio(enh_file)
    # Read clean/target signal
    clean = read_audio(tgt_file)

    # Check that both files are the same length, otherwise computing the metrics results in an error
    if len(clean) != len(enh):
        raise Exception(
            f"Wav files {enh_file} and {tgt_file} should have the same length"
        )

    # Compute binaural metrics
    m_mbstoi = compute_mbstoi(clean, enh, cfg["objective_metrics"]["fs"])

    # Store scene metrics in a tmp file
    with open(scene_metrics_file, "w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([scene_name, m_mbstoi])

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def compute_metrics(cfg: DictConfig) -> None:
    # paths to data
    enhanced = os.path.join(cfg["enhanced"])
    target = os.path.join(cfg["target"])
    # json file with info about scenes
    scenes_eval = json.load(open(cfg["scenes_names"]))
    # csv file to store metrics
    create_dir(cfg["metrics_results"])
    metrics_file = os.path.join(cfg["metrics_results"], "objective_metrics.csv")
    csv_lines = ["scene", "mbstoi"]
    
    futures = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for scene in scenes_eval:
            futures.append(executor.submit(run_metrics, scene, enhanced, target, cfg))
        proc_list = [future.result() for future in tqdm(futures)]

    # Store results in one file
    with open(metrics_file, "w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(csv_lines)
        for scene in tqdm(scenes_eval):
            scene_name = scene["scene"]
            scene_metrics_file = os.path.join(cfg["metrics_results"], f"{scene_name}.csv")
            with open(scene_metrics_file, newline='') as csv_f:
                scene_metrics = csv.reader(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in scene_metrics:
                    csv_writer.writerow(row)
            # remove tmp file
            os.system(f"rm {scene_metrics_file}")

if __name__ == "__main__":

    compute_metrics()
