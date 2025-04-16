# 🌸 Menstrual Cycle Prediction using Deep Learning & XGBoost

This project is an intelligent menstrual cycle prediction system designed to empower users with accurate, personalized insights into their reproductive health. It leverages **Deep Learning (LSTM)** to model time-based cycle patterns and **XGBoost** for symptom-driven analysis — combining both into a powerful hybrid model.

The system predicts the next period dates, ovulation window, and provides health insights based on user inputs like past cycle dates and symptoms. Built with care for usability and privacy, this solution also includes a chatbot, visual analytics, and a clean user interface via Streamlit.

---

## 🧠 Key Features

- 🔮 **Predict Next 3 Period Dates** based on cycle history
- 💡 **Ovulation Window Prediction**
- 📈 **Cycle Insights** – Regularity score, variation tracking, and symptom analytics
- 🚨 **Health Alerts** – Irregular cycle detection, personalized tips, confidence scores
- 🗣️ **Help Chatbot** – Menstrual health queries & cycle predictions
- 📊 **UI with Streamlit** – Calendar view, cycle heatmaps, and symptom graphs
- ✨ **Hybrid ML Model** – Combines LSTM (time series) & XGBoost (symptom analysis)

---

## 🧬 Technologies Used

- 🧠 **Deep Learning**: LSTM (Keras/TensorFlow)
- 🚀 **Machine Learning**: XGBoost
- 🖥️ **Frontend**: Streamlit
- 💬 **Chatbot**: Built-in rule-based logic + NLP support
- 📊 **Visualization**: Matplotlib, Seaborn
- 🗂️ **Data Handling**: Pandas, NumPy
- 🛠️ **Hybrid Model Integration**

---

## 📁 Project Structure

menstrual-cycle-predictor/ │ ├── data/ # User cycle data (CSV or JSON) ├── models/ # Saved LSTM & XGBoost models ├── notebooks/ # Training & evaluation notebooks ├── chatbot/ # Help chatbot logic ├── utils/ # Preprocessing, feature engineering ├── app/ # Streamlit app interface │ └── app.py ├── requirements.txt # Python dependencies └── README.md




## ⚙️ How to Run the Project

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/menstrual-cycle-predictor.git
cd menstrual-cycle-predictor


###Create a Virtual Environment
bash:python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

###Install Dependencies
bash:pip install -r requirements.txt

###Run the Streamlit App
cd app
streamlit run app.py



📅 Sample Inputs
Start Date: 2025-03-01

Cycle Length: 28

Symptom History: cramps, mood swings, headaches




🤖 How It Works
LSTM Model: Trained on historical cycle dates to forecast future periods.

XGBoost: Analyzes symptoms and lifestyle patterns for personalized adjustments.

Hybrid Optimizer: Combines outputs using a scoring strategy to increase accuracy.

Cycle Confidence Score: Tells you how reliable each prediction is.



📊 UI Features (Streamlit)
📆 Calendar View: Predicted periods & ovulation days

🔥 Heatmaps & Graphs: Cycle length trends, symptom severity

🧠 Chatbot Assistant: Ask “When is my next period?” or “What does mood swing mean?”

🧘‍♀️ Health Tips: Based on irregularity and symptoms



📈 Future Improvements
🔁 Real-time feedback learning for continuous personalization

📱 Mobile app support

🌐 Multilingual support for wider accessibility

🧬 Integration with wearable devices (e.g., Fitbit, Oura Ring)



🤝 Contributing
Pull requests are welcome! If you’d like to contribute with new features, fixes, or data improvements:
bash:fork → clone → create branch → commit → pull request


🙋‍♀️ Need Help?
Feel free to open an issue or reach out with feedback.
Your health matters. Let's build smarter tools together 💖
