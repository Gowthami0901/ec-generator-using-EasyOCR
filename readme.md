# 🏡 Tamil Nadu Encumbrance Certificate Generator

This project is a full-stack automation tool to generate **Encumbrance Certificates (EC)** from the [Tamil Nadu Registration Portal](https://tnreginet.gov.in). It combines:

- 🖥️ **Streamlit frontend** for user interaction
- ⚙️ **FastAPI backend** to handle PDF generation
- 🤖 **Selenium automation** to fill the TNREGINET form and download the EC PDF

---

## 📦 Features

- Select Zone, District, SRO, and Village via dropdowns
- Input EC period, survey number, and subdivision
- Automatically fills the TNREGINET portal and downloads the EC PDF
- Preview and download the generated EC in the Streamlit app

---

### 🧪 Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
streamlit
requests
selenium
fastapi
uvicorn
pydantic
```

### 🌐 Chrome + ChromeDriver

- Download [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your browser version
- Add it to your system PATH or place it in the project root

---

## 🗂️ Project Structure

```
📁 ec-generator/
├── app.py              # Streamlit UI
├── main.py             # FastAPI app
├── automation.py       # Selenium logic
├── requirements.txt    # Python dependencies
└── README.md           # Project instructions
```

---

## 🚀 How to Run

### 1. Start FastAPI backend

```bash
uvicorn main:app --reload
```

This will run the FastAPI server at `http://localhost:8000`.

---

### 2. Run Streamlit frontend

```bash
streamlit run app.py
```

This will open the Streamlit interface in your browser.

---

## 📝 Notes

- Ensure a stable internet connection — the TNREGINET portal is dynamic and may sometimes lag.
- Chrome will open during the process unless `--headless` is enabled in `automation.py`.
