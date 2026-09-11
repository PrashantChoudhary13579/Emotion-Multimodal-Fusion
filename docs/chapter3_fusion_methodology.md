# **Multimodal Fusion --- Early, Mid, and Late Fusion Experiments**

## **3.1 Overview**

This chapter documents a separate, self-contained pipeline built to
train and evaluate audio + video emotion classifiers using pretrained
self-supervised embeddings --- Whisper / Wav2Vec2 for audio and V-JEPA2
for video --- combined through three different fusion strategies.

Unlike the EmotionLLaMA v2 work in Chapter 2, this pipeline uses
lightweight classifier heads trained from scratch on top of frozen
pretrained embeddings, rather than fine-tuning a large multimodal LLM.
This makes it dramatically cheaper to train and iterate on, at the cost
of not producing the natural-language reasoning EmotionLLaMA v2 provides
--- the two approaches are complementary rather than competing, and this
chapter\'s experiments were also useful as a fast, controlled tested for
questions (e.g. \"does training on more diverse data improve
cross-dataset generalization?\") that would be far more expensive to
iterate on with a full multimodal LLM.

Target classes across all experiments: anger, disgust, fear, happy, sad.
A sixth class, neutral, exists in some raw data but was excluded from
training and evaluation throughout, to keep the label space consistent
across all four source datasets used later in this chapter.

## **3.2 Fusion Strategies Implemented**

+----------+-------------------+--------------------------------------+
| **St     | **Script**        | **Approach**                         |
| rategy** |                   |                                      |
+==========+===================+======================================+
| Early    | train_baseline.py | Concatenates audio + video           |
| fusion   |                   | embeddings into a single vector, fed |
|          | train_ba          | through one MLP (FusionMLP).         |
|          | seline_cv_fold.py |                                      |
+----------+-------------------+--------------------------------------+
| Mid      | trai              | Projects audio and video into a      |
| fusion   | n_gated_fusion.py | shared space; a learned gate weights |
|          |                   | each modality per sample before      |
|          | train_gated_      | fusion (GatedFusionClassifier).      |
|          | fusion_cv_fold.py |                                      |
+----------+-------------------+--------------------------------------+
| Late     | tra               | Trains separate classifier heads per |
| fusion   | in_late_fusion.py | modality, then combines their output |
|          |                   | logits using a learned/searched      |
|          | train_late_       | blend weight, alpha                  |
|          | fusion_cv_fold.py | (LateFusionHeads).                   |
+----------+-------------------+--------------------------------------+

At the architectural level,

-   Early fusion is the simplest and fastest to train but gives the
    model no explicit way to weigh one modality against the other --- in
    practice this tends to let whichever modality has the
    higher-dimensional or more discriminative embedding (typically
    video) dominate the classifier\'s decisions.

-   Mid (gated) fusion addresses this directly with a learned,
    per-sample gate, at the cost of a slightly larger parameter count
    and a training pipeline that requires precomputed embeddings rather
    than raw audio/video.

-   Late fusion is the most conservative design --- because each
    modality is classified fully independently before any combination
    happens, it is the easiest to debug and degrades gracefully if one
    modality is missing or unreliable at inference time, but it also has
    the least capacity to learn cross-modal interactions, since audio
    and video features are never seen together until the final logit
    blend.

![](media/image2.jpeg){width="6.268055555555556in"
height="3.4159722222222224in"}

## **3.3 Pipeline Architecture**

The end-to-end pipeline flows as follows:

-   Raw videos (with embedded audio) are split into separate audio
    (audio_extraction.py) and video streams.

-   Audio/video pairs are matched by label and filename stem across the
    Audio/ and Video/ directory trees.

-   Audio and video embeddings are extracted (and optionally
    precomputed/cached) using the chosen pretrained backbones ---

-   ![](media/image3.png){width="6.575694444444444in"
    height="4.9847222222222225in"}e.g. openai/whisper-small for audio

-   ![](media/image4.jpeg){width="6.268055555555556in"
    height="2.8375in"}facebook/vjepa2-vitl-fpc64-256 for video.

-   The cached/extracted embeddings are fed into one of the three fusion
    trainers (early / mid / late) to produce a checkpoint
    (multimodal_classifier.pt, gated_fusion.pt, or late_fusion.pt).

  -------------------------------------------------------------------------------
  **Modality**   **Model**         **Raw feature shape**  **Pooled embedding**
  -------------- ----------------- ---------------------- -----------------------
  Audio          Whisper Small     \[B, 1500, 768\]       \[B, 768\]

  Video          V-JEPA2 ViT-L     \[B, 8192, 1024\]      \[B, 1024\]
  -------------------------------------------------------------------------------

-   Trained checkpoints are evaluated using the matching evaluation
    script, producing predictions.csv, metrics.json, confusion matrices,
    and per-class F1 plots.

-   A trained checkpoint can also be run on a single new audio+video
    sample via inference.py for single-sample predictions.

A discipline maintained throughout this pipeline --- and one of the more
important lessons of this stage of the internship --- is that the
\--audio_model, \--video_model, and audio source (separately recorded
vs. extracted from video) used at evaluation time must always be matched
exactly to what was used during training.

## **3.4 Training and Cross-Dataset Testing --- Single-Dataset Baseline (SAMVEDNA-only)**

The first phase of experimentation trained the model on the SAMVEDNA
dataset only (audio and video kept in separate, paired directory trees),
and then evaluated it for generalization on unseen datasets ---
PUMAVE-D, eNTERFACE'05, and RAVDESS --- as a cross-dataset testing
setup. This tests whether a fusion model trained on one dataset\'s
recording conditions, speakers, and label conventions can generalize to
audio-visual emotion data collected under different conditions.

### **Example training run --- Early Fusion on SAMVEDNA**

The following command was used to train the early-fusion (FusionMLP)
baseline on the SAMVEDNA dataset, using Whisper-small for audio
embeddings and V-JEPA2 for video embeddings, with per-epoch evaluation
on the held-out test split enabled:

python train_baseline.py \\

\"\.../SAMVEDNA (AUDIO_VIDEO_SEPERATED)/Audio\" \\

\"\.../SAMVEDNA (AUDIO_VIDEO_SEPERATED)/Video\" \\

\--audio_model \"openai/whisper-small\" \\

\--video_model \"facebook/vjepa2-vitl-fpc64-256\" \\

\--out_dir \"\.../Multimodal_emotion results/new\" \\

\--audio_cache \"\.../.cache/audio_emb_whisper_small\" \\

\--video_cache \"\.../.cache/video_emb_vjepa2\" \\

\--epochs 20 \--batch_size 32 \--eval_test_each_epoch

This run trains for 20 epochs with a batch size of 32, computing
validation/test metrics after every epoch. The resulting checkpoint
(multimodal_classifier.pt) and its metrics.json/labels.json were saved
to the specified out_dir for later evaluation against the cross-dataset
test sets. Analogous commands (with train_gated_fusion.py and
train_late_fusion.py, run on precomputed embedding caches) were used for
the mid-fusion and late-fusion variants, following the same
SAMVEDNA-trained, cross-dataset-tested methodology.

### **Cross-dataset evaluation results (SAMVEDNA-trained)**

Once trained on SAMVEDNA alone, checkpoints were evaluated against
PUMAVE-D, eNTERFACE'05, and RAVDESS using the matching evaluation
scripts. This first round of testing exposed the central problem that
motivated the next phase of the work: a model trained on a single
dataset generalizes poorly to data collected under different recording
conditions, even though the label taxonomy is identical.

Early-fusion checkpoints trained the same way generalized considerably
worse still --- cross-dataset accuracy for the early-fusion baseline
fell as low as roughly 29--50% depending on the target dataset, compared
to a near-98% in-domain accuracy on SAMVEDNA itself. The gap between
strong in-domain performance and weak cross-dataset generalization, due
to the domain shift is exactly the problem that motivated building a
combined, multi-source training set --- described next.

## **3.5 Mixed-Dataset Training and Cross-Dataset Evaluation**

To address the generalization gap observed above, a second phase of
experimentation trained each fusion strategy on a purpose-built Mixed
Dataset that combines samples from all four source datasets, rather than
on SAMVEDNA alone. The hypothesis was straightforward: if the model
never sees more than one recording condition / speaker population during
training, it has no reason to learn features that are robust to a change
in recording condition. Training on a mixture of sources forces exactly
that.

### **3.5.1 Mixed Dataset construction**

The Mixed Dataset was built by drawing a balanced number of samples per
emotion from each of the four source datasets:

  -----------------------------------------------------------------------
  **Source dataset**        **Samples per emotion**
  ------------------------- ---------------------------------------------
  SAMVEDNA                  200

  PUMAVE-D                  200

  RAVDESS                   200

  eNTERFACE'05              200
  -----------------------------------------------------------------------

-   800 samples per emotion class (200 from each of the 4 sources)

-   5 emotion classes × 800 samples = 4,000 video clips, plus a matched
    4,000 audio files with the same class distribution

-   Split into 2,400 training / 400 validation / 1,200 test samples

Every fusion strategy (early, mid, late) was retrained from scratch on
this Mixed Dataset, using the same Whisper-small (audio) +
V-JEPA2-vitl-fpc64-256 (video) embedding backbones as the single-dataset
baseline in §3.4, so that any change in cross-dataset performance can be
attributed to the training data mixture rather than to a different
embedding model.

### **3.5.2 Training accuracy and loss curves**

All of these, early, mid (gated) and late fusion were trained on the
Mixed Dataset with per-epoch logging of training loss/accuracy alongside
validation and held-out test metrics, which allows the convergence
behaviour of each architecture to be compared directly. Selected epochs
from each run are reported below rather than the full per-epoch log, to
keep the table readable; the pattern is consistent across the epochs in
between.

### **Early fusion ---**

### **Training** 

Early fusion is trained on train_baseline_cv_fold.py file by using the
following command:

python train_baseline_cv_fold.py \\

\"\.../Mixed_Audio/Audio\" \\

\"\.../Mixed_Video/Video\" \\

\--audio_model \"openai/whisper-small\" \\

\--video_model \"facebook/vjepa2-vitl-fpc64-256\" \\

\--out_dir \"\.../Multimodal_emotion results/new\" \\

\--audio_cache \"\.../.mixed_cache/audio_emb_whisper_small\" \\

\--video_cache \"\.../.mixed_cache/video_emb_vjepa2\" \\

\--epochs 150 ---lr 1e-4 \--dropout 0.3 \\

![](media/image5.png){width="3.5131944444444443in"
height="2.9270833333333335in"}\--batch_size 32 \--eval_test_each_epoch

![](media/image6.png){width="3.421527777777778in"
height="2.5590277777777777in"}![](media/image7.png){width="3.3472222222222223in"
height="2.5208333333333335in"}

### **Testing**

![](media/image8.png){width="2.8256944444444443in"
height="2.421527777777778in"}![](media/image9.png){width="2.7909722222222224in"
height="2.3854166666666665in"} SAMVEDNA RAVDESS

![](media/image10.png){width="2.9270833333333335in"
height="2.5069444444444446in"}![](media/image11.png){width="2.998611111111111in"
height="2.56875in"} PUMAVE-D eNTERFACE\'05

-   **Implementation Details**

### For the implementation of early fusion we take raw video and extracted Audios and then fixed-length embeddings are formed using pretrained backbones --- audio via openai/whisper-small and video via facebook/vjepa2-vitl-fpc64-256 --- with the resulting embeddings cached to disk (audio_emb_whisper_small, video_emb_vjepa2) to avoid recomputation across folds.

### The resulting network is trained end-to-end for 150 epochs with a learning rate of 1e-4 using AdamW optimizer, dropout of 0.3, and a batch size of 32, using train_baseline_cv_fold.py under a cross-validation fold where k = 5, with test-set metrics logged every epoch. The model is implemented in PyTorch and conducted training and inference on 24GB NVIDIA RTX A5500 GPU with the training time nearly 3hrs.

### The results show that Early Fusion achieves the **highest accuracy of 98.88% on the MIXED dataset**, indicating that the model is highly effective when information from different modalities is combined. On **PUMAVE-D**, an accuracy of **96.89%** is obtained, while the **SAMVEDNA dataset** achieves **93.49%** accuracy. For **eNTERFACE'05**, the model achieves **93.00%**, whereas the lowest performance is observed on **RAVDESS**, with an accuracy of **89.11%**.

  ------------------------------------------------------------------------------------
  **Dataset\\Parameters**   **Accuracy**   **Precision**   **Recall**   **F1-Score**
  ------------------------- -------------- --------------- ------------ --------------
  PUMAVE-D                  96.90%         96.94%          96.89%       96.89%

  eNTERFACE'05              93.00%         93.00%          93.01%       92.99%

  RAVDESS                   89.11%         89.38%          89.11%       88.98%

  SAMVEDNA                  93.50%         93.54%          93.50%       93.46%
  ------------------------------------------------------------------------------------

###

###

###

###

###

###

### **Mid (gated) fusion ---**

### **Training**

Mid fusion is trained on train_gated_fusion_cv_fold.py file by using the
following command:

python train_gated_fusion_cv_fold.py \\

\"\.../.mixed_cache/audio_emb_whisper_small\" \\

\"\.../.mixed_cache/video_emb_vjepa2\" \\

\--audio_model \"openai/whisper-small\" \\

\--video_model \"facebook/vjepa2-vitl-fpc64-256\" \\

\--out_dir \"\.../Multimodal_emotion results/new\" \\

\--epochs 150 ---lr 2e-5 \--dropout 0.4 ---hidden 256 \\

\--batch_size 32 \--eval_test_each_epoch

![](media/image12.png){width="3.5541666666666667in"
height="2.8569444444444443in"}![](media/image13.png){width="3.5833333333333335in"
height="2.7944444444444443in"}

###

### ![](media/image14.png){width="4.145138888888889in" height="3.4541666666666666in"}**Testing**

![](media/image15.png){width="3.59375in"
height="3.078472222222222in"}![](media/image16.png){width="3.489006999125109in"
height="2.9895833333333335in"} SAMVEDNA RAVDESS

![](media/image17.png){width="3.5194444444444444in"
height="3.015277777777778in"}![](media/image18.png){width="3.509027777777778in"
height="3.00625in"} PUMAVE-D eNTERFACE\'05

### **Implementation** 

### The same cached audio and video embeddings are used, but rather than concatenating them directly, a learned gating mechanism with a hidden dimension of 256 first projects each modality\'s embedding into a shared space, then computes a data-dependent weight (gate) controlling how much each modality contributes to the fused representation before it is passed to the classification head. This lets the model down-weight a noisier or less informative modality on a per-sample basis, rather than treating both modalities as equally reliable at every input, as early fusion implicitly does.

### Because the gating parameters add another layer of optimization difficulty, this configuration is trained with a lower learning rate of 2e-5 using AdamW optimizer and a higher dropout of 0.4 (vs. 0.3 for early fusion) for 150 epochs, batch size 32, using train_gated_fusion_cv_fold.py under a cross-validation fold where k = 5, with test-set metrics logged every epoch. The model is implemented in PyTorch and conducted training and inference on 24GB NVIDIA RTX A5500 GPU with the training time nearly 3hrs.

### The **Mid Fusion** approach was evaluated across five datasets: **MIXED, PUMAVE-D, eNTERFACE'05, RAVDESS, and SAMVEDNA**. In this approach, the features from different modalities are processed separately and then combined at an intermediate stage of the model.

### Mid Fusion achieves an accuracy of **97.81% on MIXED** and **96.80% on PUMAVE-D**, showing strong performance on both datasets. The highest performance on an individual dataset is observed on **eNTERFACE'05, with an accuracy of 95.42%**. For **SAMVEDNA**, the model achieves **92.81%**, while **RAVDESS records the lowest accuracy of 88.12%**.

###

###

###

###

  ------------------------------------------------------------------------------------
  **Dataset\\Parameters**   **Accuracy**   **Precision**   **Recall**   **F1-Score**
  ------------------------- -------------- --------------- ------------ --------------
  PUMAVE-D                  96.80%         96.81%          96.82%       96.81%

  eNTERFACE'05              95.43%         95.48%          95.43%       95.45%

  RAVDESS                   89.13%         88.42%          88.13%       87.89%

  SAMVEDNA                  92.81%         92.83%          92.81%       92.79%
  ------------------------------------------------------------------------------------

###

###

###

###

###

###

### **Late fusion ---**

### **Training curve** 

Late fusion is trained on train_gated_fusion_cv_fold.py file by using
the following command:

python train_late_fusion_cv_fold.py \\

\"\.../.mixed_cache/audio_emb_whisper_small\" \\

\"\.../.mixed_cache/video_emb_vjepa2\" \\

\--audio_model \"openai/whisper-small\" \\

\--video_model \"facebook/vjepa2-vitl-fpc64-256\" \\

\--out_dir \"\.../Multimodal_emotion results/new\" \\

\--epochs 150 ---lr 1e-4 \--dropout 0.2 \--hidden 256 \\

\--batch_size 32 \--eval_test_each_epoch

![](media/image19.png){width="3.2777777777777777in"
height="2.441666666666667in"}![](media/image20.png){width="3.1944444444444446in"
height="2.3965277777777776in"}

###

###

###

###

### ![](media/image21.png){width="3.787037401574803in" height="3.156003937007874in"}**Testing** 

### ![](media/image22.png){width="3.138888888888889in" height="2.689583333333333in"}![](media/image23.png){width="3.1756944444444444in" height="2.720833333333333in"}SAMVEDNA RAVDESS

### ![](media/image24.png){width="3.2237292213473316in" height="2.762037401574803in"}![](media/image25.png){width="3.2194444444444446in" height="2.7583333333333333in"}PUMAVE-D eNTERFACE\'05

### **Implementation**

### Audio and video embeddings are again taken from the same cached representations, but each modality is passed through its own separate branch (with hidden dimension 256) that produces an independent per-modality prediction; these modality-specific outputs are only combined at the final decision stage, e.g., by averaging or a lightweight learned combination of the two branches\' logits.

### Cross-modal interaction is therefore minimal compared to early or gated fusion --- the model largely learns unimodal audio and video predictors that are reconciled only at the end. This configuration uses a learning rate of 1e-4 using AdamW optimizer and dropout of 0.2, trained for 150 epochs with batch size 32 via train_late_fusion_cv_fold.py under a cross-validation fold where k = 5, with test-set metrics logged every epoch. The model is implemented in PyTorch and conducted training and inference on 24GB NVIDIA RTX A5500 GPU with the training time nearly 3hrs.

### The Late Fusion approach was evaluated on **MIXED, PUMAVE-D, eNTERFACE'05, RAVDESS, and SAMVEDNA** datasets. It achieved an accuracy of **98.75% on MIXED**, **97.48% on PUMAVE-D**, and **96.08% on eNTERFACE'05**. On **RAVDESS**, the model obtained **93.12%**, while **SAMVEDNA** achieved **95.08%** accuracy.

  ------------------------------------------------------------------------------------
  **Dataset\\Parameters**   **Accuracy**   **Precision**   **Recall**   **F1-Score**
  ------------------------- -------------- --------------- ------------ --------------
  PUMAVE-D                  97.48%         97.50%          97.48%       97.48%

  eNTERFACE'05              96.08%         96.19%          96.09%       96.10%

  RAVDESS                   93.13%         93.21%          93.13%       93.10%

  SAMVEDNA                  95.08%         95.10%          95.08%       95.06%
  ------------------------------------------------------------------------------------

All the architectures show the expected pattern: rapid loss reduction
and accuracy gain in the first \~25--50 epochs while the model learns
the coarse class boundaries, followed by a longer regime of slow, noisy
improvement as it refines decision boundaries between
visually/acoustically similar emotion pairs (in the confusion matrices
behind these numbers, fear and disgust are consistently the
most-confused pair for both architectures). .

**3.5.3 Cross-dataset evaluation after Mixed-Dataset training**

  -------------------------------------------------------------------------------------------------
  **Fusion\\Dataset**   **MIXED**    **PUMAVE-D**   **eNTERFACE'05**   **RAVDESS**   **SAMVEDNA**
  --------------------- ------------ -------------- ------------------ ------------- --------------
  Early Fusion          98.88%       96.89%         93.00%             89.11%        93.49%

  Mid Fusion            97.81%       96.80%         95.42%             88.12%        92.81%

  Late Fusion           98.75%       97.48%         96.08%             93.12%        95.08%
  -------------------------------------------------------------------------------------------------

This is the key comparison for this chapter: the same four test sets
used in §3.4 (SAMVEDNA in-domain, plus PUMAVE-D, RAVDESS, and
eNTERFACE'05 as cross-dataset tests), now evaluated against checkpoints
trained on the Mixed Dataset instead of SAMVEDNA alone. All numbers
below are read directly from each run\'s metrics.json.

*Table: compiled cross-dataset results after training on the Mixed
Dataset. Sample sizes: SAMVEDNA n=6,750; PUMAVE-D n=1,032; RAVDESS
n=1,920; eNTERFACE'05 n=1,072. Late-fusion cross-dataset test results
were not present in the current experiment archive for the
Mixed-Dataset-trained checkpoint (the training/testing commands used are
recorded, but the corresponding output metrics.json files were not
saved/retained) --- this row is intentionally left out rather than
estimated, and should be filled in once that evaluation is (re-)run.*

### **3.5.4 Discussion: single-dataset vs. mixed-dataset training**

Comparing the three fusion techniques across the Mixed Dataset clearly
shows that the choice of fusion strategy affects performance differently
across datasets:

• **Early Fusion** achieved **98.88% on MIXED, 96.89% on PUMAVE-D,
93.00% on eNTERFACE'05, 89.11% on RAVDESS, and 93.49% on SAMVEDNA**. It
performed particularly well on the MIXED and PUMAVE-D datasets but
showed comparatively lower generalization on RAVDESS.

• **Mid Fusion** obtained **97.81% on MIXED, 96.80% on PUMAVE-D, 95.42%
on eNTERFACE'05, 88.12% on RAVDESS, and 92.81% on SAMVEDNA**. Although
its performance was slightly lower on MIXED, PUMAVE-D, RAVDESS, and
SAMVEDNA compared with Early Fusion, it achieved better performance on
eNTERFACE'05.

• **Late Fusion** provided the most consistent overall performance,
achieving **98.75% on MIXED, 97.48% on PUMAVE-D, 96.08% on eNTERFACE'05,
93.12% on RAVDESS, and 95.08% on SAMVEDNA**. It obtained the **highest
accuracy among the three techniques on eNTERFACE'05, RAVDESS, and
SAMVEDNA**.

• Comparing the overall results, **Late Fusion demonstrates the
strongest cross-dataset performance**, particularly on RAVDESS and
SAMVEDNA, where it outperforms both Early and Mid Fusion by a noticeable
margin. This suggests that combining modality-specific predictions at a
later stage can provide better robustness across datasets with different
characteristics.

The practical takeaway is that **Late Fusion performed the most
consistently across the evaluated datasets**, while Early Fusion
achieved the highest accuracy on the MIXED dataset. Mid Fusion showed
competitive performance but did not consistently outperform the other
approaches. Overall, the results suggest that **late-stage integration
of modality-specific information can improve the model\'s ability to
generalize across diverse emotion-recognition datasets**.

# **Chapter 4:**
