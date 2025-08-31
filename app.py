import streamlit as st
import requests
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
from io import BytesIO
import matplotlib.pyplot as plt

# กำหนดค่าหน้า
st.set_page_config(
    page_title="Image Processing App",
    page_icon="🖼️",
    layout="wide"
)

# หัวข้อหลัก
st.title("🖼️ Image Processing Application")
st.markdown("แอปพลิเคชันสำหรับประมวลผลภาพ พร้อม GUI ที่ใช้งานง่าย")

# Sidebar สำหรับ navigation
st.sidebar.title("🛠️ เครื่องมือ")
option = st.sidebar.selectbox(
    "เลือกฟีเจอร์ที่ต้องการใช้งาน",
    ["📥 อัพโหลดภาพ", "🌐 ดาวน์โหลดจาก URL", "🎨 ประมวลผลภาพ", "📊 กราฟและข้อมูล"]
)

# ฟังก์ชันสำหรับโหลดภาพจาก URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"ไม่สามารถโหลดภาพได้: {e}")
        return None

# ฟังก์ชันสำหรับ custom image processing
def apply_custom_filter(image, filter_type, intensity=1.0):
    """Apply custom filters to image"""
    if filter_type == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=intensity))
    elif filter_type == "Sharpen":
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1 + intensity)
    elif filter_type == "Brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(intensity)
    elif filter_type == "Contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(intensity)
    elif filter_type == "Saturation":
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(intensity)
    elif filter_type == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_type == "Edge Detection":
        return image.filter(ImageFilter.FIND_EDGES)
    elif filter_type == "Invert":
        return ImageOps.invert(image.convert('RGB'))
    else:
        return image

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# หน้าอัพโหลดภาพ
if option == "📥 อัพโหลดภาพ":
    st.header("📥 อัพโหลดภาพจากคอมพิวเตอร์")
    
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพ",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="รองรับไฟล์ PNG, JPG, JPEG, GIF, BMP"
    )
    
    if uploaded_file is not None:
        # โหลดและแสดงภาพ
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ภาพต้นฉบับ")
            st.image(image, caption="ภาพที่อัพโหลด", use_container_width=True)
        
        with col2:
            st.subheader("ข้อมูลภาพ")
            st.write(f"**ชื่อไฟล์:** {uploaded_file.name}")
            st.write(f"**ขนาด:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**โหมดสี:** {image.mode}")
            st.write(f"**ขนาดไฟล์:** {len(uploaded_file.getvalue())} bytes")

# หน้าดาวน์โหลดจาก URL
elif option == "🌐 ดาวน์โหลดจาก URL":
    st.header("🌐 ดาวน์โหลดภาพจาก URL")
    
    url = st.text_input(
        "ใส่ URL ของภาพ",
        placeholder="https://example.com/image.jpg",
        help="ใส่ลิงก์ภาพที่ต้องการดาวน์โหลด"
    )
    
    if st.button("📥 ดาวน์โหลดภาพ"):
        if url:
            with st.spinner("กำลังดาวน์โหลดภาพ..."):
                image = load_image_from_url(url)
                if image:
                    st.session_state.current_image = image
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("ภาพที่ดาวน์โหลด")
                        st.image(image, caption="ภาพจาก URL", use_container_width=True)
                    
                    with col2:
                        st.subheader("ข้อมูลภาพ")
                        st.write(f"**URL:** {url}")
                        st.write(f"**ขนาด:** {image.size[0]} x {image.size[1]} pixels")
                        st.write(f"**โหมดสี:** {image.mode}")
        else:
            st.warning("กรุณาใส่ URL ของภาพ")

# หน้าประมวลผลภาพ
elif option == "🎨 ประมวลผลภาพ":
    st.header("🎨 ประมวลผลภาพ")
    
    if st.session_state.current_image is None:
        st.warning("กรุณาอัพโหลดภาพก่อน หรือดาวน์โหลดจาก URL")
        st.info("ไปที่หน้า 'อัพโหลดภาพ' หรือ 'ดาวน์โหลดจาก URL' ก่อน")
    else:
        # แสดงภาพต้นฉบับ
        st.subheader("ภาพต้นฉบับ")
        st.image(st.session_state.current_image, caption="ภาพต้นฉบับ", width=400)
        
        # Custom Parameters สำหรับ Image Processing
        st.subheader("🛠️ การปรับแต่งภาพ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # เลือกประเภทฟิลเตอร์
            filter_type = st.selectbox(
                "เลือกฟิลเตอร์",
                ["None", "Blur", "Sharpen", "Brightness", "Contrast", "Saturation", "Emboss", "Edge Detection", "Invert"],
                help="เลือกประเภทฟิลเตอร์ที่ต้องการใช้"
            )
            
            # ปรับความเข้มของฟิลเตอร์
            if filter_type in ["Blur", "Sharpen", "Brightness", "Contrast", "Saturation"]:
                if filter_type == "Blur":
                    intensity = st.slider("ความเข้มของ Blur", 0.1, 5.0, 1.0, 0.1)
                elif filter_type == "Brightness":
                    intensity = st.slider("ความสว่าง", 0.1, 3.0, 1.0, 0.1)
                elif filter_type == "Contrast":
                    intensity = st.slider("ความคมชัด", 0.1, 3.0, 1.0, 0.1)
                elif filter_type == "Saturation":
                    intensity = st.slider("ความอิ่มตัวสี", 0.0, 3.0, 1.0, 0.1)
                else:
                    intensity = st.slider("ความเข้ม", 0.1, 3.0, 1.0, 0.1)
            else:
                intensity = 1.0
        
        with col2:
            # การปรับขนาด
            st.subheader("📏 ปรับขนาดภาพ")
            resize_option = st.checkbox("ปรับขนาดภาพ")
            
            if resize_option:
                original_width, original_height = st.session_state.current_image.size
                
                new_width = st.number_input("ความกว้าง (pixels)", 
                                          min_value=50, 
                                          max_value=2000, 
                                          value=original_width)
                
                new_height = st.number_input("ความสูง (pixels)", 
                                           min_value=50, 
                                           max_value=2000, 
                                           value=original_height)
        
        # ปุ่มประมวลผล
        if st.button("🚀 ประมวลผลภาพ", type="primary"):
            with st.spinner("กำลังประมวลผลภาพ..."):
                processed_image = st.session_state.current_image.copy()
                
                # ใช้ฟิลเตอร์
                if filter_type != "None":
                    processed_image = apply_custom_filter(processed_image, filter_type, intensity)
                
                # ปรับขนาด
                if resize_option:
                    processed_image = processed_image.resize((int(new_width), int(new_height)))
                
                st.session_state.processed_image = processed_image
                
                # แสดงผลลัพธ์
                st.success("ประมวลผลเสร็จสิ้น!")
        
        # แสดงภาพที่ประมวลผลแล้ว
        if st.session_state.processed_image is not None:
            st.subheader("ภาพหลังประมวลผล")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.processed_image, 
                        caption="ภาพหลังประมวลผล", 
                        use_container_width=True)
            
            with col2:
                # ข้อมูลภาพใหม่
                st.write("**ข้อมูลภาพใหม่:**")
                st.write(f"ขนาด: {st.session_state.processed_image.size[0]} x {st.session_state.processed_image.size[1]}")
                st.write(f"โหมดสี: {st.session_state.processed_image.mode}")
                
                # ดาวน์โหลดภาพ
                buf = BytesIO()
                st.session_state.processed_image.save(buf, format="PNG")
                
                st.download_button(
                    label="💾 ดาวน์โหลดภาพ",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

# หน้ากราฟและข้อมูล
elif option == "📊 กราฟและข้อมูล":
    st.header("📊 การวิเคราะห์ภาพและกราฟ")
    
    if st.session_state.current_image is None:
        st.warning("กรุณาอัพโหลดภาพก่อน")
    else:
        # แปลงภาพเป็น numpy array
        image_array = np.array(st.session_state.current_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Histogram ของสีแดง-เขียว-น้ำเงิน")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if len(image_array.shape) == 3:  # ภาพสี
                colors = ['red', 'green', 'blue']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
                    ax.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
            else:  # ภาพขาวดำ
                hist = cv2.calcHist([image_array], [0], None, [256], [0, 256])
                ax.plot(hist, color='black', label='Grayscale')
            
            ax.set_xlabel('ค่าความเข้มของสี')
            ax.set_ylabel('จำนวน Pixels')
            ax.set_title('Color Histogram')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("📋 สถิติภาพ")
            
            # คำนวณสถิติ
            if len(image_array.shape) == 3:
                # ภาพสี
                mean_rgb = np.mean(image_array, axis=(0, 1))
                std_rgb = np.std(image_array, axis=(0, 1))
                
                st.write("**ค่าเฉลี่ยสี (RGB):**")
                st.write(f"🔴 Red: {mean_rgb[0]:.2f}")
                st.write(f"🟢 Green: {mean_rgb[1]:.2f}")
                st.write(f"🔵 Blue: {mean_rgb[2]:.2f}")
                
                st.write("**ส่วนเบียงเบนมาตรฐาน (RGB):**")
                st.write(f"🔴 Red: {std_rgb[0]:.2f}")
                st.write(f"🟢 Green: {std_rgb[1]:.2f}")
                st.write(f"🔵 Blue: {std_rgb[2]:.2f}")
            else:
                # ภาพขาวดำ
                mean_gray = np.mean(image_array)
                std_gray = np.std(image_array)
                
                st.write(f"**ค่าเฉลี่ย:** {mean_gray:.2f}")
                st.write(f"**ส่วนเบียงเบนมาตรฐาน:** {std_gray:.2f}")
            
            st.write(f"**ค่าสูงสุด:** {np.max(image_array)}")
            st.write(f"**ค่าต่ำสุด:** {np.min(image_array)}")
            st.write(f"**จำนวน Pixels:** {image_array.size:,}")
        
        # เปรียบเทียบก่อนและหลังประมวลผล
        if st.session_state.processed_image is not None:
            st.subheader("📊 เปรียบเทียบก่อนและหลังประมวลผล")
            
            processed_array = np.array(st.session_state.processed_image)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram ภาพต้นฉบับ
            if len(image_array.shape) == 3:
                colors = ['red', 'green', 'blue']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
                    ax1.plot(hist, color=color, alpha=0.7)
            
            ax1.set_title('ภาพต้นฉบับ')
            ax1.set_xlabel('ค่าความเข้มของสี')
            ax1.set_ylabel('จำนวน Pixels')
            ax1.grid(True, alpha=0.3)
            
            # Histogram ภาพที่ประมวลผลแล้ว
            if len(processed_array.shape) == 3:
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([processed_array], [i], None, [256], [0, 256])
                    ax2.plot(hist, color=color, alpha=0.7)
            
            ax2.set_title('ภาพหลังประมวลผล')
            ax2.set_xlabel('ค่าความเข้มของสี')
            ax2.set_ylabel('จำนวน Pixels')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 คำแนะนำการใช้งาน")
st.sidebar.markdown("""
1. **อัพโหลดภาพ** - เลือกไฟล์จากคอมพิวเตอร์
2. **ดาวน์โหลดจาก URL** - ใส่ลิงก์ภาพ
3. **ประมวลผลภาพ** - ใช้ฟิลเตอร์และปรับแต่ง
4. **ดูกราฟและข้อมูล** - วิเคราะห์ภาพ
""")

st.sidebar.markdown("### ℹ️ เกี่ยวกับแอป")
st.sidebar.info("แอปพลิเคชันนี้สร้างด้วย Streamlit สำหรับการประมวลผลภาพแบบง่ายๆ")

# แสดง session state ปัจจุบัน (สำหรับ debug)
if st.sidebar.checkbox("🔧 แสดงสถานะ Debug"):
    st.sidebar.write("Current image loaded:", st.session_state.current_image is not None)
    st.sidebar.write("Processed image available:", st.session_state.processed_image is not None)
