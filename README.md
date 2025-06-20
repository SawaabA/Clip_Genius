# 🏆 Scoreboard Analyzer AI

An AI-powered sports video analyzer built during the **Google Datathon @ Laurier Analytics 2025**, focused on transforming long, under-promoted sports footage into short, high-impact highlight reels—automatically.

---

## 🎯 Problem Addressed

Lesser-known sports leagues and women’s sports often lack visibility due to time-consuming editing processes and limited coverage. Gen Z, in particular, demands **short, engaging, and shareable content**—something current workflows struggle to deliver.

---

## ✨ Our Solution

**Scoreboard Analyzer AI** automates the entire video-editing pipeline for sports highlights by combining:
- 🧠 Computer Vision (OpenCV)
- 🧾 OCR (PyTesseract)
- 🔊 NLP (NVIDIA NeMo)
- ⚡ FFmpeg-based parallel processing

Together, they detect scoreboard events, analyze commentary for excitement, and auto-generate highlight clips within seconds.

---

## 🚀 Key Features

- **🎥 Real-Time Score Tracking**  
  Extracts scoreboard values from frames using edge detection and OCR.

- **🗣️ Commentary Analysis with NLP**  
  Ranks moments based on audio excitement using NeMo.

- **💡 Highlight Clip Generation**  
  Automatically generates clips around key score events.

- **⚡ Efficient Processing**  
  Uses multi-threaded FFmpeg processing to speed up analysis (saving ~14 hours per project).

---

## ⚙️ Tech Stack

- `OpenCV` – Video frame processing  
- `PyTesseract` – OCR for score extraction  
- `FFmpeg` – Clip generation and parallelization  
- `NeMo (NVIDIA)` – Natural Language Processing for commentary  
- `FAISS` – For embedding and similarity search  
- `Python threading` – Multi-threaded segment processing

---

## 🧪 Usage

```bash
python main.py <video_path> --function <function_name> --debug --output <output_dir>


## 🙌 Contributors

- **JD** – [jashandeep.co.uk](https://jashandeep.co.uk)  
- **Robert Pevec** – [robertpevec.com](https://robertpevec.com)  
- **Sawaab Anas** – [GitHub](https://github.com/SawaabA)  
- **Suhana Khullar** – [Instagram](https://instagram.com)

---

## 🧠 Acknowledgments

The **Laurier Analytics Hackathon** was an incredible opportunity to collaborate, innovate, and push boundaries in a short timeframe. Over the course of the event, our team worked relentlessly—designing, building, and overcoming technical hurdles while gaining valuable new skills.

We are especially thankful to the **Laurier Analytics team** and **Google Waterloo** for organizing this event and creating a platform for creative problem-solving and growth.

> 🙏 **Special thanks to our mentor, Shivam Garg**, for offering technical guidance and thoughtful feedback that helped us polish the project and tackle challenges head-on.

---

## 🔄 Next Iteration Plan

- 🔍 **Upgrade OCR Engine**  
  Integrate advanced OCR tools like **Google Vision API** or **AWS Textract** for more accurate score recognition.

- 🎯 **Improve Accuracy**  
  Fine-tune score detection logic to reduce false positives and improve reliability across diverse footage.

- ☁️ **Cloud Integration**  
  Enable cloud-based video processing pipelines with scalable storage (e.g., Google Cloud, AWS S3).

- 📦 **Robust Storage Layer**  
  Implement database and media storage systems to persist analysis results, clips, and metadata.
