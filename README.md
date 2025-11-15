<div align="center">

<h2 align="center">
Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment (AAAI 2026)
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2511.10334">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2511.10334-b31b1b.svg" alt="Paper">
  </a>
  <a href=""> 
    <img src="https://img.shields.io/badge/Code-Coming_Soon-orange.svg" alt="Code">
  </a>
</p>

</div>

> [!IMPORTANT]
> **Code will be released soon. Stay tuned!**

> **Abstract:** Recent advancements in weakly-supervised video anomaly detection have achieved remarkable performance by applying the multiple instance learning paradigm based on multimodal foundation models such as CLIP to highlight anomalous instances and classify categories.
However, their objectives may tend to detect the most salient response segments, while neglecting to mine diverse normal patterns separated from anomalies, and are prone to category confusion due to similar appearance, leading to unsatisfactory fine-grained classification results.
Therefore, we propose a novel Disentangled Semantic Alignment Network (DSANet) to explicitly separate abnormal and normal features from coarse-grained and fine-grained aspects, enhancing the distinguishability.
Specifically, at the coarse-grained level, we introduce a self-guided normality modeling branch that reconstructs input video features under the guidance of learned normal prototypes, encouraging the model to exploit normality cues inherent in the video, thereby improving the temporal separation of normal patterns and anomalous events.
At the fine-grained level, we present a decoupled contrastive semantic alignment mechanism, which first temporally decomposes each video into event-centric and background-centric components using frame-level anomaly scores and then applies visual-language contrastive learning to enhance class-discriminative representations.
Comprehensive experiments on two standard benchmarks, namely XD-Violence and UCF-Crime, demonstrate that DSANet outperforms existing state-of-the-art methods. 


## 📜 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@misc{yin2025learningtellapartweakly,
      title={Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment}, 
      author={Wenti Yin and Huaxin Zhang and Xiang Wang and Yuqing Lu and Yicheng Zhang and Bingquan Gong and Jialong Zuo and Li Yu and Changxin Gao and Nong Sang},
      year={2025},
      eprint={2511.10334},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.10334}, 
}
