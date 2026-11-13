<p align="center">
  <img src=".github/assets/banner.svg" alt="sbi21cm: optimal information compression for inference on the 21-cm signal from the Epoch of Reionisation" width="100%">
</p>

<p align="center">
  <a href="#-citation"><img src="https://img.shields.io/badge/paper-A%26A%20(in%20prep.)-1d4e89" alt="paper"></a>
  <img src="https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A5%202.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch >= 2.0">
  <img src="https://img.shields.io/badge/nflows-0.14-6f42c1" alt="nflows 0.14">
  <img src="https://img.shields.io/badge/data-LoReLi%20II-0b1d3a" alt="LoReLi II">
  <img src="https://img.shields.io/badge/noise-SKA--Low%20100%20%7C%2050%20%7C%2025%20h-e07a1f" alt="SKA-Low noise">
</p>

<p align="center">
  <a href="#-key-results"><b>Results</b></a> ·
  <a href="#-the-pipeline"><b>Pipeline</b></a> ·
  <a href="#-quick-start"><b>Quick start</b></a> ·
  <a href="#-reproducing-the-paper"><b>Reproduce the paper</b></a> ·
  <a href="#-data-format"><b>Data</b></a> ·
  <a href="#-citation"><b>Cite</b></a>
</p>

---

This repository contains the code for the paper

> **Optimal information compression for inference on the 21 cm signal from the Epoch of Reionisation**<br>
> Dilruwan, Benoit, David — LUX, Observatoire de Paris, Université PSL .. <br>
> *Astronomy & Astrophysics*, in preparation

The 21-cm signal from the Epoch of Reionisation is usually analysed through hand-crafted statistics: the power spectrum (PS), and more recently its combination with the L-moments of the pixel distribution function (PDF+PS). These statistics capture only part of the non-Gaussian structure of the field. We ask whether **a 3D convolutional network trained to maximise the mutual information between its summary and the astrophysical parameters (VMIM)** extracts more information from SKA-like noisy observations than PDF+PS, **without degrading the calibration of the posterior**.



---

<p align="center"><sub>LUX · Observatoire de Paris · Université PSL </sub></p>
