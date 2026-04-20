# 🚀 Quick Start Guide - 2026 Global Trend Forecaster

## ⚡ 30-Second Setup

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Train Models
```bash
python train_model.py
```

You should see output like:
```
✅ Dataset loaded: 100 records, 5 features
✅ Vectorizer created with 100 features
✅ Engagement Model trained: R² = 0.72
✅ Sentiment Model trained: Accuracy = 0.85
✅ All models saved to 'models.pkl'
```

#### Step 3: Run the App
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📱 Using the App

### Home Page 🏠
- Read about the project
- See dataset overview
- Learn about key features

### Trend Analytics 📊
- Explore trending topics
- View sentiment distribution
- Analyze platform engagement
- See regional trends
- Check word clouds

### Real-time Forecasting 🔮
1. Enter a headline
2. Select platform (X, News, Google)
3. Choose region (Global, India, USA)
4. Click "Analyze Trend Power"
5. View predictions:
   - Engagement Score (0-100)
   - Sentiment (Positive/Neutral/Negative)
   - Detailed recommendations

---

## 📂 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app (850+ lines) |
| `train_model.py` | Model training script |
| `trending_topics_2026.csv` | Dataset (100 records) |
| `models.pkl` | Trained models (auto-generated) |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `setup.bat` | Windows setup script |
| `setup.sh` | Mac/Linux setup script |
| `QUICKSTART.md` | This file |

---

## 🎯 Sample Predictions

### Test Headline 1: Technology
**Input:**
- Headline: "AI breakthrough revolutionizes healthcare"
- Platform: X
- Region: Global

**Expected Output:**
- Engagement: ~85/100 🔥 VIRAL
- Sentiment: 😊 POSITIVE

### Test Headline 2: Climate
**Input:**
- Headline: "Global warming accelerates worldwide"
- Platform: News
- Region: Global

**Expected Output:**
- Engagement: ~75/100 📈 HIGH
- Sentiment: 😢 NEGATIVE

### Test Headline 3: Finance
**Input:**
- Headline: "Stock market shows mixed signals"
- Platform: News
- Region: USA

**Expected Output:**
- Engagement: ~60/100 ➡️ MEDIUM
- Sentiment: 😐 NEUTRAL

---

## ✅ What's Included

✨ **Complete Working Application**
- Home page with overview
- Trend analytics with 6+ visualizations
- Real-time forecasting with detailed analysis

🤖 **ML Models**
- Random Forest (Engagement Score prediction)
- Logistic Regression (Sentiment Classification)
- TF-IDF Text Vectorization

📊 **Visualizations**
- Word clouds
- Pie charts (sentiment)
- Bar charts (platform engagement)
- Histograms (distribution)
- Engagement meter

🎨 **Professional UI**
- Custom CSS styling
- Color-coded outputs
- Responsive design
- Interactive elements

📚 **Documentation**
- Comprehensive README
- Inline code comments
- Quick start guide
- API documentation

---

## 🐛 Common Issues & Solutions

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: models.pkl"
```bash
python train_model.py
```

### "Port 8501 is already in use"
```bash
streamlit run app.py --server.port 8502
```

### Models training fails
- Ensure `trending_topics_2026.csv` is in the same directory
- Check that all dependencies are installed
- Python 3.7+ is recommended

---

## 🎓 For Academic Submission

This project includes:
- ✅ Clean, well-documented code
- ✅ Professional UI/UX design
- ✅ Working ML models
- ✅ Rich visualizations
- ✅ Complete documentation
- ✅ Error handling
- ✅ Ready-to-run setup scripts

**Ready for grading!** Just run `setup.bat` (Windows) or `./setup.sh` (Mac/Linux)

---

## 💡 Tips & Tricks

### Try These Headlines:
- "Quantum computing breakthrough in 2026"
- "Global economic crisis intensifies"
- "New renewable energy innovation unveiled"
- "Tech company announces major layoffs"
- "Climate action plan approved by UN"

### Experiment With:
- Different platforms (see how X vs News differs)
- Different regions (Global vs India vs USA)
- Long vs short headlines
- Positive vs negative language
- Technical vs non-technical topics

---

## 📞 Need Help?

1. Read the **README.md** for detailed documentation
2. Check inline **code comments** in app.py
3. Review the **Model Training Output** for diagnostics
4. Test with the **Sample Headlines** provided above

---

## 🚀 Next Steps

After running the app:

1. ✅ Explore the Home page
2. ✅ Check Trend Analytics visualizations
3. ✅ Try forecasting with sample headlines
4. ✅ Experiment with different inputs
5. ✅ Review the code and model details
6. ✅ Submit for academic evaluation

---

**Status**: Production Ready ✅
**Version**: 1.0
**Last Updated**: April 2026

Enjoy predicting global trends! 🌍📊🚀
