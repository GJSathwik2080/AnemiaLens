# 🩺 AnemiaLens Web App — Quick Start Guide

## ✅ Everything is Ready!

Your beautiful, fully functional web app is ready to use. Follow the steps below to launch it.

---

## 🚀 How to Run the Web App

### Option 1: Quick Start (Recommended)

Open your terminal in the project folder and run:

```bash
# Activate the environment
.\anemia_env\Scripts\Activate.ps1

# Run the web app
streamlit run app.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

### Option 2: With Custom Port

If port 8501 is busy, use a different port:

```bash
streamlit run app.py --server.port=8502
```

---

## 📖 How to Use the Web App

### Step-by-Step Instructions:

1. **App Opens in Browser**
   - A beautiful, gradient-designed interface loads
   - You'll see clear instructions at the top

2. **Upload 3 Images** (All required)
   - 👁️ **Eye (Conjunctiva)**: Inner eyelid area
   - 💅 **Nail**: Nail bed close-up
   - 🤚 **Palm**: Palm area of your hand
   
   Tips for best results:
   - Use **good natural lighting**
   - Images should be **clear and in focus**
   - **Close-up** photos work better
   - Supported formats: PNG, JPG, JPEG

3. **Click "Analyze All Images"**
   - The button activates once all 3 images are uploaded
   - AI models will process each image
   - Real-time progress indicators show status

4. **Review Beautiful Results**
   - **Individual Predictions**: Color-coded scores for each body part
   - **Combined Score**: Weighted average from all 3 organs
   - **Final Diagnosis**: 🟢 Healthy or 🔴 Anemic (with severity)
   - **Color Key**:
     - 🟢 **Green** = Healthy Signs
     - 🟡 **Yellow** = Mild Anemic Signs
     - 🟠 **Orange** = Moderate Anemic Signs
     - 🔴 **Red** = Severe Anemic Signs

5. **Important Next Steps**
   - If positive result: Schedule a blood test (CBC)
   - Always consult a healthcare professional
   - This is a screening tool, not a diagnosis

---

## 🎨 What You'll See

### Beautiful UI Features:

✨ **Gradient-based Design**
- Purple to Blue gradient headers
- Green (Healthy) and Orange (Anemic) result cards
- Professional color-coded status indicators

📊 **Clear Data Visualization**
- Large percentage displays for each organ
- Weighted score breakdown (50% Eye, 30% Nail, 20% Palm)
- Progress indicators during analysis
- Severity level badges (Mild/Moderate/Severe)

⚠️ **Important Disclaimers**
- Clearly marked as screening tool only
- Emphasized need for blood test confirmation
- Medical disclaimer always visible

📋 **Next Steps Guidance**
- Personalized recommendations based on results
- Symptoms to watch for
- Instructions for positive/negative outcomes

---

## 🔧 Troubleshooting

### **Port Already in Use**
```bash
# Use a different port
streamlit run app.py --server.port=8502
```

### **Models Not Loading**
Make sure you've trained the models first:
```bash
python main.py
```

### **Image Processing Error**
- Upload high-quality, clear images
- Ensure good lighting in photos
- Try different image formats (JPG/PNG)

### **Slow Performance**
- The first prediction takes longer (model loading)
- Subsequent predictions are faster (cached)
- Normal processing time: 30-60 seconds per analysis

---

## 📊 Model Weights Explained

The combined score is calculated as:
```
Combined Score = (Eye × 50%) + (Nail × 30%) + (Palm × 20%)
```

**Why these weights?**
- **Conjunctiva (50%)**: Most sensitive visual indicator
- **Nail (30%)**: Clear visual changes in anemia
- **Palm (20%)**: Secondary indicator, supporting evidence

---

## ⚕️ Important Medical Disclaimer

🚨 **This application is a screening tool ONLY**

- **NOT a medical diagnosis**
- **NOT a substitute** for blood tests (ABC, CBC)
- **MUST be confirmed** with professional healthcare provider
- **Seek immediate help** if experiencing:
  - Severe shortness of breath
  - Chest pain
  - Extreme fatigue
  - Loss of consciousness

---

## 🎯 Command Reference

```bash
# Activate environment
.\anemia_env\Scripts\Activate.ps1

# Run web app (default port 8501)
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port=8502

# Run in headless mode (no browser)
streamlit run app.py --headless=true

# Test with sample images (terminal version)
python predict.py
```

---

## 📁 Project Files Structure

```
e:\ml_projects\health\
├── app.py                      ← WEB APP (Streamlit)
├── predict.py                  ← Command-line prediction
├── data_pipeline.py            ← Data processing
├── model_builder.py            ← Model architecture
├── trainer.py                  ← Training logic
├── segmentation.py             ← ROI isolation
├── saved_models/
│   ├── conjunctiva_final.keras
│   ├── nail_final.keras
│   └── palm_final.keras
├── test_images/                ← Sample test images
└── anemia_env/                 ← Virtual environment
```

---

## 🧪 Testing the Web App

### Test with Sample Images:

1. Run the web app: `streamlit run app.py`
2. Click on each upload box
3. Browse to `test_images/` folder
4. Upload eye.png, nail.png, palm.png
5. Click "Analyze All Images"
6. View detailed results

---

## 📞 Support

If you encounter issues:

1. **Verify models are trained**
   ```bash
   python main.py
   ```

2. **Check environment is activated**
   ```bash
   .\anemia_env\Scripts\Activate.ps1
   ```

3. **Verify streamlit is installed**
   ```bash
   pip install streamlit==1.28.0 pillow
   ```

4. **Check image files exist**
   ```bash
   # For command-line testing
   python predict.py test_images/eye.png test_images/nail.png test_images/palm.png
   ```

---

## 🎉 You're All Set!

Your AnemiaLens web app is ready to use. Start the app and enjoy the beautiful interface!

```bash
streamlit run app.py
```

Made with ❤️ for healthcare accessibility.
