import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from transformers import EncodecModel, AutoProcessor
import torchaudio
from encodec.utils import convert_audio
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
    parser.add_argument("--nq", type=int)
    parser.add_argument("--in_scp", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--model_path", type=str, default="/home/neo.dzh/src/encodec_32khz")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = EncodecModel.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = model.to(args.device)

    for line in open(args.in_scp, "rt"):
        uttid, file_path = line.strip().split(" ", maxsplit=1)
        wav, sr = torchaudio.load(file_path)
        # wav = wav[0].to(args.device)
        inputs = processor(
            raw_audio=wav[0],
            sampling_rate=processor.sampling_rate,
            return_tensors="pt"
        )
        inputs["input_values"] = convert_audio(
            inputs["input_values"], sr,
            model.config.sampling_rate, model.config.audio_channels
        )
        inputs["padding_mask"] = torch.ones_like(inputs["input_values"], dtype=torch.int32)[0]
        with torch.no_grad():
            encoder_outputs = model.encode(
                inputs["input_values"].to(args.device),
                inputs["padding_mask"].to(args.device)
            )
            recon_wav = model.decode(
                encoder_outputs.audio_codes[:, :, :args.nq, :],
                encoder_outputs.audio_scales,
                inputs["padding_mask"]
            )[0]
        recon_wav = recon_wav.squeeze(0).cpu()
        recon_wav = convert_audio(recon_wav, model.config.sampling_rate, sr, 1)
        out_path = os.path.join(args.out_dir, f"{uttid}.wav")
        recon_wav = recon_wav.numpy()[0]
        soundfile.write(out_path,
                        (recon_wav * (2 ** 15)).astype(np.int16),
                        sr, "PCM_16", "LITTLE", "WAV", True)
        logging.info(f"Process {uttid}, length: {recon_wav.shape[0]}.")
