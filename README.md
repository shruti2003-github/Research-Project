# 🍽️ Mumbai Street Food Classifier

A deep learning web app that classifies **6 Mumbai street food items** using **MobileNetV2** (Transfer Learning), achieving **91.33% test accuracy**.

**Food categories:** Vada Pav · Sandwich · Samosa · Pani Puri · Dosa · Idli

---

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Prepare your repository

Your repo should look like this:
```
your-repo/
├── app.py
├── model.h5          ← your trained model (upload this!)
├── requirements.txt
└── README.md
```

### Step 2 — Upload model.h5
Since `model.h5` can be large, use **Git LFS**:
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add model.h5
git commit -m "Add model"
git push
```
> Alternatively, host the model on Google Drive and load it via URL in `app.py`.

### Step 3 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository → set **Main file path** to `app.py`
4. Click **Deploy** 🎉

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Performance

| Model       | Accuracy | Precision | Recall | F1-score |
|-------------|----------|-----------|--------|----------|
| MobileNetV2 | **91.33%** | 0.91 | 0.90 | 0.90 |
| VGG16       | 87.30%   | 0.87 | 0.87 | 0.87 |
| ResNet50    | 27.00%   | 0.27 | 0.27 | 0.27 |

---

## 🗂️ Dataset
- 2,400 images (400 per class)
- Preprocessing: resize to 224×224, normalization
- Augmentation: rotation, flipping, zoom, colour jitter
- Split: 70% train / 15% validation / 15% test

---

## 📄 Research
*Comparative Analysis of Deep Learning Models for Mumbai Street Food Image Recognition*  
**Shruti Kesharwani** · B. K. Birla College, Kalyan, Maharashtra  
Guided by Mr. Kalpesh Gaikwad (Assistant Professor, Dept. of IT)
