# Streamlit-app
# 🖼️ Image Processing Application

แอปพลิเคชันประมวลผลภาพที่สร้างด้วย Streamlit พร้อม GUI ที่ใช้งานง่าย รองรับการอัพโหลดภาพ การดาวน์โหลดจาก URL และการประมวลผลภาพด้วยฟิลเตอร์ต่างๆ

## ✨ ฟีเจอร์หลัก

### 📥 การอัพโหลดภาพ
- รองรับไฟล์ภาพหลากหลายรูปแบบ: PNG, JPG, JPEG, GIF, BMP
- แสดงข้อมูลภาพแบบละเอียด (ขนาด, โหมดสี, ขนาดไฟล์)
- อินเทอร์เฟซแบบ drag-and-drop

### 🌐 ดาวน์โหลดภาพจาก URL
- ดาวน์โหลดภาพจาก internet ผ่าน URL
- ตรวจสอบความถูกต้องของ URL อัตโนมัติ
- การจัดการข้อผิดพลาดที่ครอบคลุม

### 🎨 ประมวลผลภาพขั้นสูง
- **ฟิลเตอร์ภาพ 9 ประเภท:**
  - 🌫️ Blur - เบลอภาพ
  - ✨ Sharpen - เพิ่มความคมชัด
  - ☀️ Brightness - ปรับความสว่าง
  - 🔲 Contrast - ปรับความคมชัด
  - 🌈 Saturation - ปรับความอิ่มตัวสี
  - 🎭 Emboss - เอฟเฟกต์นูน
  - 📐 Edge Detection - ตรวจจับขอบ
  - 🔄 Invert - กลับสี

- **การปรับแต่งแบบ Custom:**
  - Slider สำหรับปรับความเข้มของฟิลเตอร์
  - การปรับขนาดภาพแบบ manual
  - พรีวิวผลลัพธ์แบบ real-time

### 📊 การวิเคราะห์ภาพ
- **Color Histogram:** แสดงการกระจายตัวของสี RGB
- **สถิติภาพ:** ค่าเฉลี่ย, ส่วนเบียงเบนมาตรฐาน, ค่าสูงสุด-ต่ำสุด
- **การเปรียบเทียบ:** กราฟเปรียบเทียบก่อนและหลังประมวลผล

### 💾 การจัดการผลลัพธ์
- แสดงภาพก่อนและหลังประมวลผลแบบเคียงข้าง
- ดาวน์โหลดภาพที่ประมวลผลแล้วในรูปแบบ PNG
- บันทึกการตั้งค่าในระหว่างเซสชัน

## 🚀 การติดตั้งและใช้งาน

### ข้อกำหนดระบบ
- Python 3.7 หรือสูงกว่า
- RAM อย่างน้อย 2GB (แนะนำ 4GB สำหรับภาพขนาดใหญ่)

### การติดตั้ง Libraries
```bash
pip install streamlit pillow opencv-python matplotlib requests numpy
```

หรือใช้ไฟล์ requirements.txt:
```bash
pip install -r requirements.txt
```

### การรันแอปพลิเคชัน
```bash
streamlit run app.py
```

แอปพลิเคชันจะเปิดใน browser ที่ `http://localhost:8501`

## 📋 วิธีการใช้งาน

### ขั้นตอนที่ 1: เตรียมภาพ
1. ไปที่หน้า **"📥 อัพโหลดภาพ"** เพื่ออัพโหลดจากคอมพิวเตอร์
2. หรือไปที่ **"🌐 ดาวน์โหลดจาก URL"** เพื่อดาวน์โหลดจาก internet

### ขั้นตอนที่ 2: ประมวลผลภาพ
1. ไปที่หน้า **"🎨 ประมวลผลภาพ"**
2. เลือกฟิลเตอร์ที่ต้องการ
3. ปรับความเข้มด้วย slider
4. กำหนดขนาดใหม่ (ถ้าต้องการ)
5. กดปุ่ม **"🚀 ประมวลผลภาพ"**

### ขั้นตอนที่ 3: วิเคราะห์ผลลัพธ์
1. ไปที่หน้า **"📊 กราฟและข้อมูล"**
2. ดู histogram และสถิติภาพ
3. เปรียบเทียบก่อนและหลังประมวลผล

### ขั้นตอนที่ 4: บันทึกผลลัพธ์
1. กดปุ่ม **"💾 ดาวน์โหลดภาพ"** ในหน้าประมวลผลภาพ
2. ไฟล์จะถูกบันทึกเป็น `processed_image.png`

## 🛠️ เทคโนโลยีที่ใช้

- **[Streamlit](https://streamlit.io/)** - Web framework สำหรับ Machine Learning และ Data Science
- **[Pillow (PIL)](https://pillow.readthedocs.io/)** - Python Imaging Library สำหรับการประมวลผลภาพ
- **[OpenCV](https://opencv.org/)** - Computer Vision library สำหรับการคำนวณ histogram
- **[Matplotlib](https://matplotlib.org/)** - Library สำหรับสร้างกราฟและการแสดงผล
- **[NumPy](https://numpy.org/)** - Library สำหรับการคำนวณทางคณิตศาสตร์
- **[Requests](https://requests.readthedocs.io/)** - HTTP library สำหรับดาวน์โหลดภาพจาก URL

## 📁 โครงสร้างโปรเจกต์

```
image-processing-app/
│
├── app.py                 # ไฟล์หลักของแอปพลิเคชัน
├── requirements.txt       # รายชื่อ dependencies
├── README.md             # เอกสารนี้
└── examples/             # โฟลเดอร์ตัวอย่างภาพ (optional)
    ├── sample1.jpg
    ├── sample2.png
    └── sample3.gif
```

## 🔧 การปรับแต่งและพัฒนาเพิ่มเติม

### การเพิ่มฟิลเตอร์ใหม่
เพิ่มฟิลเตอร์ใหม่ในฟังก์ชัน `apply_custom_filter()`:

```python
elif filter_type == "Your_New_Filter":
    # ใส่โค้ดฟิลเตอร์ใหม่ที่นี่
    return your_processed_image
```

### การปรับแต่ง UI
แก้ไขใน `st.set_page_config()` เพื่อเปลี่ยนธีมและ layout:

```python
st.set_page_config(
    page_title="Your App Name",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## ⚠️ ข้อจำกัด

- รองรับไฟล์ภาพขนาดไม่เกิน 200MB
- ฟิลเตอร์บางตัวอาจใช้เวลานานกับภาพขนาดใหญ่
- ต้องการการเชื่อมต่อ internet สำหรับการดาวน์โหลดจาก URL


## 📄 requirements.txt

```txt
streamlit>=1.28.0
Pillow>=9.0.0
opencv-python>=4.5.0
matplotlib>=3.5.0
requests>=2.25.0
numpy>=1.21.0
```

## 👨‍💻 การพัฒนา

หากต้องการพัฒนาหรือปรับปรุงแอปพลิเคชัน:

1. Fork repository นี้
2. สร้าง feature branch (`git checkout -b feature/new-feature`)
3. Commit การเปลี่ยนแปลง (`git commit -am 'Add new feature'`)
4. Push ไปยัง branch (`git push origin feature/new-feature`)
5. สร้าง Pull Request


---

**Made with ❤️ using Streamlit**
