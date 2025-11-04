"""
Green O2 Exchange - Streamlit v5 (Auto Reduce Capacity)
✅ QR scanner via webcam (auto-fill tube & branch)
✅ Upload QR image
✅ Auto form muncul setelah QR terbaca
✅ Upload Foto KTP + Selfie dengan KTP
✅ OCR otomatis dari foto KTP untuk auto-fill data diri (termasuk NIK)
✅ Data tersimpan ke data/borrow_log.csv
✅ Foto tersimpan di data/uploads/
✅ Kapasitas cabang otomatis berkurang di branches.csv
"""

import os
from datetime import datetime
from urllib.parse import parse_qs
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import re
import pytesseract
import av
from pyzbar.pyzbar import decode as zbar_decode
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase

# ---------------------------------
# SETUP PATHS
# ---------------------------------
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

BORROW_LOG = os.path.join(DATA_DIR, "borrow_log.csv")
BRANCHES_CSV = os.path.join(DATA_DIR, "branches.csv")

# ---------------------------------
# PYTESSERACT SETUP
# ---------------------------------
# Jika kamu di Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------
def ensure_borrow_log():
    if not os.path.exists(BORROW_LOG):
        df = pd.DataFrame(columns=[
            "timestamp", "tube_id", "branch", "action", "name", "nik",
            "phone", "address", "purpose", "est_duration", "return_date",
            "notes", "foto_ktp", "foto_selfie_ktp"
        ])
        df.to_csv(BORROW_LOG, index=False)

def log_borrow_row(row: dict):
    ensure_borrow_log()
    df = pd.read_csv(BORROW_LOG)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(BORROW_LOG, index=False)

def parse_tube_branch_from_text(text: str):
    if not text:
        return None, None
    text = text.strip()
    if "?" in text:
        try:
            params = parse_qs(text.split("?", 1)[1])
            tube = params.get("tube", [""])[0]
            branch = params.get("branch", [""])[0]
            return tube, branch
        except Exception:
            return None, None
    return None, None

# ---------------------------------
# FUNGSI KURANGI KAPASITAS CABANG
# ---------------------------------
def reduce_branch_capacity(branch_name):
    """Kurangi kapasitas cabang ketika ada peminjaman"""
    try:
        if not os.path.exists(BRANCHES_CSV):
            return
        df = pd.read_csv(BRANCHES_CSV)
        if branch_name in df["branch"].values:
            idx = df.index[df["branch"] == branch_name][0]
            current_capacity = df.at[idx, "capacity"]
            try:
                current_capacity = int(current_capacity)
            except ValueError:
                current_capacity = 0
            if current_capacity > 0:
                df.at[idx, "capacity"] = current_capacity - 1
                df.to_csv(BRANCHES_CSV, index=False)
                st.info(f"📉 Kapasitas cabang '{branch_name}' berkurang menjadi {current_capacity - 1}.")
            else:
                st.warning(f"⚠️ Kapasitas cabang '{branch_name}' sudah habis!")
        else:
            st.warning(f"Cabang '{branch_name}' tidak ditemukan di branches.csv")
    except Exception as e:
        st.error(f"Gagal memperbarui kapasitas: {e}")

# ---------------------------------
# OCR: BACA DATA DARI FOTO KTP
# ---------------------------------
def extract_ktp_data(image: Image.Image):
    try:
        arr = np.array(image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    except Exception:
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    try:
        gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except Exception:
        pass
    gray = cv2.medianBlur(gray, 3)

    try:
        text = pytesseract.image_to_string(gray, lang='ind')
        if not text.strip():
            text = pytesseract.image_to_string(gray, lang='eng')
    except Exception:
        text = pytesseract.image_to_string(gray, lang='eng')

    text_clean = re.sub(r'[\n\r]+', '\n', text).upper()

    nik_match = re.search(r'([0-9]{16})', text_clean)
    nik_label_match = re.search(r'N\s*I\s*K[:\s]*([0-9]{16})', text_clean)
    if nik_label_match:
        nik = nik_label_match.group(1).strip()
    elif nik_match:
        nik = nik_match.group(1).strip()
    else:
        nik = ""

    nama_match = re.search(r'NAMA[:\s]*([A-Z\s\.]+)', text_clean)
    if not nama_match:
        lines = text_clean.splitlines()
        nama_val = ""
        for i, ln in enumerate(lines):
            if 'NAMA' in ln:
                after = ln.split('NAMA', 1)[1].strip(": .")
                if after:
                    nama_val = after
                elif i + 1 < len(lines):
                    nama_val = lines[i + 1].strip()
                break
    else:
        nama_val = nama_match.group(1).strip()

    ttl_match = re.search(r'TEMPAT[/\s]*TGL\s*LAHIR[:\s]*([A-Z0-9,\s\-\/\.]+)', text_clean)
    if not ttl_match:
        ttl_match = re.search(r'TTL[:\s]*([A-Z0-9,\s\-\/\.]+)', text_clean)
    ttl_val = ttl_match.group(1).strip() if ttl_match else ""

    alamat_match = re.search(r'ALAMAT[:\s]*([A-Z0-9\s,./\-]+)', text_clean)
    if not alamat_match:
        lines = text_clean.splitlines()
        alamat_val = ""
        for i, ln in enumerate(lines):
            if 'ALAMAT' in ln:
                after = ln.split('ALAMAT', 1)[1].strip(": .")
                if after:
                    alamat_val = after
                else:
                    segs = []
                    for j in range(i+1, min(i+4, len(lines))):
                        segs.append(lines[j].strip())
                    alamat_val = " ".join([s for s in segs if s])
                break
    else:
        alamat_val = alamat_match.group(1).strip()

    return {
        "nik": nik,
        "name": nama_val.title() if nama_val else "",
        "ttl": ttl_val.title() if ttl_val else "",
        "address": alamat_val.title() if alamat_val else "",
        "raw_text": text_clean
    }

# ---------------------------------
# STREAMLIT SETUP
# ---------------------------------
st.set_page_config(layout="wide", page_title="Green O2 Exchange - Auto QR + OCR KTP")
st.title("📦 Green O₂ Exchange — QR Scanner + OCR KTP")

if "decoded_tube" not in st.session_state:
    st.session_state.decoded_tube = ""
if "decoded_branch" not in st.session_state:
    st.session_state.decoded_branch = ""
if "last_scan" not in st.session_state:
    st.session_state.last_scan = ""

# ---------------------------------
# PETA CABANG
# ---------------------------------
col_map, col_scan = st.columns([2, 1])

with col_map:
    st.header("📍 Peta Cabang")
    m = folium.Map(location=[-6.2, 106.816], zoom_start=11)
    if os.path.exists(BRANCHES_CSV):
        try:
            branches = pd.read_csv(BRANCHES_CSV)
            for _, r in branches.iterrows():
                try:
                    folium.Marker(
                        [r["lat"], r["lon"]],
                        popup=f"{r['branch']} (Kapasitas: {r.get('capacity', '-')})"
                    ).add_to(m)
                except Exception:
                    pass
        except Exception:
            st.warning("Gagal membaca branches.csv, pastikan format CSV benar.")
    else:
        st.info("File branches.csv tidak ditemukan di folder data/. Peta akan tampil tanpa marker.")
    st_folium(m, width=700, height=420)

# ---------------------------------
# SCANNER AREA
# ---------------------------------
with col_scan:
    st.header("🎥 QR Scanner")

    class QRVideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_decoded = ""

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            decoded = zbar_decode(Image.fromarray(gray))
            if decoded:
                txt = decoded[0].data.decode("utf-8")
                self.last_decoded = txt
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="qr-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=QRVideoTransformer,
        async_transform=True,
    )

    if ctx and ctx.state.playing:
        placeholder = st.empty()
        for _ in range(50):
            if ctx.video_transformer and getattr(ctx.video_transformer, "last_decoded", ""):
                txt = ctx.video_transformer.last_decoded
                if txt != st.session_state.last_scan:
                    tube, branch = parse_tube_branch_from_text(txt)
                    if tube:
                        st.session_state.decoded_tube = tube
                        st.session_state.decoded_branch = branch or ""
                        st.session_state.last_scan = txt
                        placeholder.success(f"✅ QR Terdeteksi: {tube} / {branch}")
                    else:
                        placeholder.warning(f"⚠️ QR tidak valid: {txt}")
                break

    st.markdown("---")

    uploaded = st.file_uploader("📤 Atau upload gambar QR (opsional)", type=["png", "jpg", "jpeg"])
    if uploaded:
        try:
            pil = Image.open(uploaded).convert("RGB")
            decoded = zbar_decode(pil)
            if decoded:
                txt = decoded[0].data.decode("utf-8")
                tube, branch = parse_tube_branch_from_text(txt)
                if tube:
                    st.session_state.decoded_tube = tube
                    st.session_state.decoded_branch = branch or ""
                    st.success(f"✅ QR dari gambar: {tube} / {branch}")
                else:
                    st.error("❌ QR tidak mengandung tube/branch.")
            else:
                st.error("❌ Tidak ada QR terdeteksi.")
        except Exception as e:
            st.error(f"Error membaca file QR: {e}")

# ---------------------------------
# FORM PEMINJAMAN
# ---------------------------------
st.markdown("---")
if st.session_state.decoded_tube:
    st.subheader("🧾 Form Peminjaman Tabung")

    with st.form("borrow_form"):
        col1, col2 = st.columns(2)
        with col1:
            tube_id = st.text_input("Tube ID", value=st.session_state.decoded_tube, disabled=True)
            branch = st.text_input("Branch", value=st.session_state.decoded_branch or "", disabled=True)

            st.markdown("### 📸 Upload Foto")
            foto_ktp = st.file_uploader("Unggah Foto KTP (otomatis dibaca OCR)", type=["jpg", "jpeg", "png"])
            foto_selfie = st.file_uploader("Unggah Foto Selfie dengan KTP", type=["jpg", "jpeg", "png"])

            auto_data = {"nik": "", "name": "", "address": ""}

            if foto_ktp:
                st.image(foto_ktp, caption="Preview KTP", width=250)
                try:
                    pil_img = Image.open(foto_ktp).convert("RGB")
                    extracted = extract_ktp_data(pil_img)
                except Exception as e:
                    extracted = {"nik": "", "name": "", "address": "", "raw_text": f"Error: {e}"}

                auto_data["nik"] = extracted.get("nik", "") or ""
                auto_data["name"] = extracted.get("name", "") or ""
                auto_data["address"] = extracted.get("address", "") or ""

                if auto_data["nik"] or auto_data["name"]:
                    st.success("✅ Data dari KTP berhasil dibaca otomatis!")
                    with st.expander("📋 Hasil OCR mentah"):
                        st.text(extracted.get("raw_text", ""))
                else:
                    st.warning("⚠️ Tidak bisa membaca teks dari KTP, isi manual.")

            name = st.text_input("Nama Peminjam", value=auto_data["name"])
            nik = st.text_input("NIK", value=auto_data["nik"])
            phone = st.text_input("Nomor Telepon")

        with col2:
            address = st.text_area("Alamat", value=auto_data["address"], height=100)
            purpose = st.text_input("Keperluan")
            est_duration = st.text_input("Estimasi Durasi (misal: 3 hari)")
            return_date = st.date_input("Perkiraan Tanggal Pengembalian")

        submit = st.form_submit_button("💾 Simpan")

        if submit:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            def save_upload(file, prefix):
                if file is None:
                    return ""
                filename = f"{prefix}_{timestamp}_{tube_id}.jpg"
                save_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    image = Image.open(file).convert("RGB")
                    image.save(save_path)
                    return filename
                except Exception:
                    try:
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())
                        return filename
                    except Exception:
                        return ""

            foto_ktp_path = save_upload(foto_ktp, "ktp")
            foto_selfie_path = save_upload(foto_selfie, "selfie")

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tube_id": tube_id,
                "branch": branch,
                "action": "borrow",
                "name": name,
                "nik": nik,
                "phone": phone,
                "address": address,
                "purpose": purpose,
                "est_duration": est_duration,
                "return_date": str(return_date),
                "notes": "auto-scanned",
                "foto_ktp": foto_ktp_path,
                "foto_selfie_ktp": foto_selfie_path
            }

            try:
                log_borrow_row(row)
                reduce_branch_capacity(branch)
                st.success("✅ Data peminjaman tersimpan & kapasitas cabang diperbarui.")
                st.session_state.decoded_tube = ""
                st.session_state.decoded_branch = ""
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")

# ---------------------------------
# LOG PEMINJAMAN + PREVIEW FOTO
# ---------------------------------
st.markdown("---")
st.header("📚 Log Peminjaman Terbaru")
ensure_borrow_log()
try:
    df_log = pd.read_csv(BORROW_LOG)
    st.dataframe(df_log.sort_values("timestamp", ascending=False).head(30), use_container_width=True)
except Exception as e:
    st.error(f"Gagal membaca log peminjaman: {e}")

st.subheader("🖼️ Preview Foto Terbaru")
try:
    df_log_sorted = pd.read_csv(BORROW_LOG).sort_values("timestamp", ascending=False).head(5)
    for _, row in df_log_sorted.iterrows():
        st.markdown(f"**🕒 {row.get('timestamp', '')} — {row.get('name', '')} ({row.get('tube_id', '')})**")
        col1, col2 = st.columns(2)
        with col1:
            foto_ktp_filename = row.get("foto_ktp", "")
            if isinstance(foto_ktp_filename, str) and foto_ktp_filename:
                path_ktp = os.path.join(UPLOAD_DIR, foto_ktp_filename)
                if os.path.exists(path_ktp):
                    st.image(path_ktp, caption="Foto KTP", width=250)
                else:
                    st.warning("Foto KTP tidak ditemukan.")
            else:
                st.info("Tidak ada foto KTP.")
        with col2:
            foto_selfie_filename = row.get("foto_selfie_ktp", "")
            if isinstance(foto_selfie_filename, str) and foto_selfie_filename:
                path_selfie = os.path.join(UPLOAD_DIR, foto_selfie_filename)
                if os.path.exists(path_selfie):
                    st.image(path_selfie, caption="Selfie dengan KTP", width=250)
                else:
                    st.warning("Foto selfie tidak ditemukan.")
            else:
                st.info("Tidak ada foto selfie.")
except Exception as e:
    st.error(f"Gagal menampilkan preview foto: {e}")
