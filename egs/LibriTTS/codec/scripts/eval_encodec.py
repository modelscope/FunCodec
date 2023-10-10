import os
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
import argparse
import logging
import numpy as np
import soundfile
logging.basicConfig(
    level='INFO',
    format=f"[{os.uname()[1].split('.')[0]}]"
           f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq", type=float)
    parser.add_argument("--in_scp", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = EncodecModel.encodec_model_24khz()
    model = model.to(args.device)
    model.set_target_bandwidth(args.nq)

    for line in open(args.in_scp, "rt"):
        uttid, file_path = line.strip().split(" ", maxsplit=1)
        wav, sr = torchaudio.load(file_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0).to(args.device)
        with torch.no_grad():
            encoded_frames = model.encode(wav)
            recon_wav = model.decode(encoded_frames)
        recon_wav = recon_wav.squeeze(0).cpu()
        recon_wav = convert_audio(recon_wav, model.sample_rate, sr, 1)
        out_path = os.path.join(args.out_dir, f"{uttid}.wav")
        # torchaudio.save(out_path, recon_wav, sr, encoding="PCM_S", bits_per_sample="16")
        recon_wav = recon_wav.numpy()[0]
        soundfile.write(out_path,
                        (recon_wav * (2 ** 15)).astype(np.int16),
                        sr, "PCM_16", "LITTLE", "WAV", True)
        logging.info(f"Process {uttid}, length: {recon_wav.shape[0]}.")
