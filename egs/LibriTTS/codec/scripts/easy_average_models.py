import os
import sys
import torch

if __name__ == '__main__':
    root_path = "exp/encodec_lstm_16k_n32_600k_step_rmseg_use_power_ds640_raw_en_ptts_run2/"
    idx_list = [16, 17, 18, 19, 20]
    n_models = len(idx_list)
    metrix = "visqol"

    avg = None
    for idx in idx_list:
        model_file = os.path.join(root_path, "{}epoch.pth".format(str(idx)))
        states = torch.load(model_file, map_location="cpu")
        if avg is None:
            avg = states
        else:
            for k in avg:
                avg[k] = avg[k] + states[k]

    for k in avg:
        if str(avg[k].dtype).startswith("torch.int"):
            pass
        else:
            avg[k] = avg[k] / n_models

    output_file = os.path.join(root_path, "valid.{}.ave_{}best.pth".format(metrix, n_models))
    torch.save(avg, output_file)
