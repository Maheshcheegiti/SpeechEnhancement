import streamlit as st
import os
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.enh_inference import SeparateSpeech
import soundfile
import numpy as np
from scipy import signal

ALLOWED_EXTENSIONS = {'wav'}

d = ModelDownloader()
cfg = d.download_and_unpack("espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw")
enh_model_sc = SeparateSpeech(
  train_config=cfg["train_config"],
  model_file=cfg["model_file"],
  # for segment-wise process on long speech
  normalize_segment_scale=False,
  show_progressbar=True,
  ref_channel=4,
  normalize_output_wav=True,
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file):
    speech, rate = soundfile.read(file)
    sr = 16000
    assert rate == sr, "mismatch in sampling rate"
    wave = enh_model_sc(speech[None, ...], sr)
    return speech, wave[0].squeeze(), sr

def main():
    st.set_page_config(page_title="Speech Enhancement | ESPNET", page_icon="🔊", layout="wide")

    st.title("Speech Enhancement - ESPNET 🔊")

    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            with st.spinner("Processing..."):
                speech, enhanced, sr = process_file(uploaded_file)
            st.text("Original audio")
            st.audio(speech, format='audio/wav', start_time=0, sample_rate=sr)
            st.text("Enhanced audio")
            st.audio(enhanced, format='audio/wav', start_time=0, sample_rate=sr)
        else:
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == '__main__':
    main()
