import streamlit as st
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import difflib
import librosa
import numpy as np
from dotenv import load_dotenv
import os
import tempfile
import webcolors
from PIL import Image
import io
from collections import Counter
import webcolors

load_dotenv()

st.markdown("""
<style>
.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 300px;
    background-color: black;
    color: white;
    text-align: center;
    border-radius: 7px;
    padding: 9px;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* 위쪽으로 배치 */
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>""", unsafe_allow_html=True)

# 노래 선택 state 초기화
if "songs" not in st.session_state:
    st.session_state.songs = []

if "selected_songs" not in st.session_state:
    st.session_state.selected_songs = []

if "past_selected_songs" not in st.session_state:
    st.session_state.past_selected_songs = []

if "searched" not in st.session_state:
    st.session_state.searched = False

# Hugging Face API 설정(Stable Diffusion)
#API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-3.5-large"
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Spotify API 설정
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Deezer API를 사용하여 특정 노래의 미리 듣기 URL을 가져오는 함수
def get_deezer_preview_url(song_name, artist_name):
    search_url = f"https://api.deezer.com/search?q={song_name} {artist_name}"
    
    # API 요청
    response = requests.get(search_url)
    
    # 상태 코드가 200이면 정상적인 응답
    if response.status_code == 200:
        data = response.json()
        
        # 'data' 키가 존재하고, 그 안에 곡이 있다면
        if "data" in data and len(data["data"]) > 0:
            # 제목과 아티스트의 정확한 매칭을 위해 fuzzy matching 사용
            best_match = None
            highest_ratio = 0
            
            for track in data["data"]:
                # 제목과 아티스트의 매칭 비율 계산
                title_ratio = difflib.SequenceMatcher(None, track["title"].lower(), song_name.lower()).ratio()
                artist_ratio = difflib.SequenceMatcher(None, track["artist"]["name"].lower(), artist_name.lower()).ratio()
                
                # 두 비율을 합산하여 더 높은 비율을 찾음
                total_ratio = title_ratio + artist_ratio
                
                # 가장 높은 비율을 가진 트랙 선택
                if total_ratio > highest_ratio:
                    highest_ratio = total_ratio
                    best_match = track
            
            if best_match:
                return best_match["preview"]
    
    return None  # Deezer에서 결과가 없을 경우

# 오디오 다운로드 및 librosa로 특징 추출
def extract_audio_features(url):
    """Deezer MP3를 librosa로 직접 분석"""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        st.warning(f"오디오 다운로드 실패 (status code: {response.status_code}) | URL: {url}")
        raise ValueError("오디오 다운로드 실패!")

    # 임시 파일에 MP3 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_filename = tmp_file.name  # 파일 경로 저장
        tmp_file.write(response.content)

    try:
        tmp_file.close()

        # librosa로 MP3 파일 로드
        audio_data, sr = librosa.load(tmp_filename, sr=None)

        # 오디오 특징 추출
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        if tempo > 170:
            tempo /= 2 
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)

        features = {
            'tempo': tempo,
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth)
        }
        return features

    finally:
        # 파일 삭제
        os.remove(tmp_filename)

# 게이지와 툴팁 생성 함수
def gauge_with_tooltip(value, label, min_val, max_val, tooltip_text):
    percent = (value - min_val) / (max_val - min_val) * 100
    percent = max(0, min(percent, 100))

    st.markdown(f"""
    <div class="tooltip" style="margin-bottom: 14px; width: 260px; font-size: 15px;">
        <b>{label}: {value:.2f}</b>
        <span class="tooltiptext" style="width: 240px; padding: 6px;">
            {tooltip_text}
        </span>
        <div style="background-color: #eee; border-radius: 7px; height: 16px; width: 260px;">
            <div style="height: 100%; width: {percent:.1f}%; background-color: #ff4b4b; border-radius: 7px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 종합적인 분위기 계산
def aggregate_features(features_list):
    if not features_list:
        return None
    avg_features = {key: np.mean([f[key] for f in features_list]) for key in features_list[0]}

    st.write("### 플레이리스트 분석 결과")
    st.markdown(f"🎵 평균 템포: {avg_features['tempo']:.2f} BPM")

    gauge_with_tooltip(
        avg_features['spectral_centroid'],
        "🎶 평균 스펙트럴 센트로이드",
        min_val=1000,
        max_val=7000,
        tooltip_text="이 값이 높을수록 음색이 밝습니다."
    )

    gauge_with_tooltip(
        avg_features['spectral_bandwidth'],
        "🎸 평균 스펙트럴 밴드위드",
        min_val=500,
        max_val=4500,
        tooltip_text="이 값이 클수록 음악의 다이내믹 레인지가 넓습니다."
    )

    return avg_features

# 헥스 색상을 이름으로 변환
CSS3_NAMES_TO_HEX = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "lime": "#00ff00", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff", "silver": "#c0c0c0", "gray": "#808080",
    "maroon": "#800000", "olive": "#808000", "green": "#008000", "purple": "#800080", "teal": "#008080",
    "navy": "#000080", "orange": "#ffa500", "pink": "#ffc0cb", "brown": "#a52a2a", "gold": "#ffd700",
    "beige": "#f5f5dc", "coral": "#ff7f50", "turquoise": "#40e0d0", "violet": "#ee82ee"
}
def hex_to_color_name(hex_color):
    try:
        return webcolors.hex_to_name(hex_color)
    except ValueError:
        def closest_color(hex_value):
            r, g, b = webcolors.hex_to_rgb(hex_value)
            closest_name = None
            min_distance = float("inf")
            for name, hex_code in CSS3_NAMES_TO_HEX.items():
                rc, gc, bc = webcolors.hex_to_rgb(hex_code)
                distance = (r - rc) ** 2 + (g - gc) ** 2 + (b - bc) ** 2
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
            return closest_name
        return closest_color(hex_color)

# Stable Diffusion 이미지 생성
def generate_playlist_image(features, style, color, seed_mode):
    color_name = hex_to_color_name(color)
    target_rgb = webcolors.hex_to_rgb(color)

    # 프롬프트 구성
    if style == "Color":
        style_prompt = (
            f"The entire image MUST be a smooth and uninterrupted gradient using ONLY rich and saturated shades of {color_name}. "
            "Do NOT include any characters, objects, or identifiable shapes. "
            "The image must be abstract, minimal, and focused purely on color. "
            f"{color_name} must completely dominate the image. No other colors are allowed."
        )

    elif style == "Character":
        style_prompt = (
            "Place a full-body Japanese anime style character clearly in the center of the image. "
            "The character MUST reflect the emotional mood of the music through pose, outfit, and expression. "
            f"The background MUST be a smooth gradient in saturated {color_name} tones. "
            "The background must be clean and simple, with no distracting elements. "
            f"The lighting and color tone of the entire image must be influenced by {color_name}."
        )

    elif style == "Landscape":
        style_prompt = (
            "Create a visually compelling landscape that reflects the mood of the music. "
            f"The landscape must feature a strong presence of {color_name} tones throughout the sky, ground, or water. "
            f"Use only natural elements (mountains, clouds, rivers, trees, etc.) in harmony with {color_name}. "
            "Avoid cityscapes, buildings, or modern elements."
        )

    elif style == "Abstract":
        style_prompt = (
            "Design a fully abstract composition that powerfully conveys the emotional tone of the music. "
            f"The color {color_name} MUST dominate the entire image. "
            "No recognizable objects or scenes should be visible. "
            "Use textures, forms, and brush-like patterns to evoke feeling and intensity through color and structure."
        )


    prompt = "A playlist cover reflecting the overall musical vibe:"

    # ✅ 음악적 특징 분석 포함
    if features['tempo'] > 160:
        prompt += " A very fast and high-energy track, often found in intense rock or electronic music."
    elif features['tempo'] > 130:
        prompt += " A fast and energetic rhythm, commonly heard in rock, punk, and dance music."
    elif features['tempo'] > 100:
        prompt += " A moderately fast tempo, giving a vibrant and lively feel."
    elif features['tempo'] > 70:
        prompt += " A balanced rhythm with a relaxed yet engaging pace."
    else:
        prompt += " A slow and soothing track with a calm and peaceful atmosphere."

    if features['spectral_centroid'] > 5500:
        prompt += " A bright and sharp sound, often associated with high-energy rock and metal."
    elif features['spectral_centroid'] > 4000:
        prompt += " A slightly bright yet warm tone, commonly found in pop rock and alternative music."
    elif features['spectral_centroid'] > 2500:
        prompt += " A well-balanced sound with a mix of warmth and clarity."
    else:
        prompt += " A deep and mellow tone, often associated with acoustic and jazz music."

    if features['spectral_bandwidth'] > 3000:
        prompt += " A highly dynamic and expressive sound with a wide frequency range."
    elif features['spectral_bandwidth'] > 2000:
        prompt += " A vibrant and energetic texture, often found in rock and upbeat tracks."
    elif features['spectral_bandwidth'] > 1200:
        prompt += " A smooth and clear sound with a mix of mellow and bright elements."
    else:
        prompt += " A soft and warm sound with subtle variations, ideal for calm and acoustic music."

    prompt += " " + style_prompt

    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2))

    def get_dominant_color(img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((32, 32))
        pixels = list(img.getdata())
        most_common = Counter(pixels).most_common(1)[0][0]
        return most_common

    if seed_mode == "고정된 시드로 1장 생성":
        payload = {"inputs": prompt, "parameters": {"seed": 42}}
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            st.write(prompt)
            return response.content
        else:
            st.error("이미지 생성 실패!")
            return None
    else:
        seeds = [11, 22, 33, 44, 55]
        best_img = None
        best_score = float("inf")
        st.write("여러 이미지 중 최적의 결과를 선택합니다...")

        for seed in seeds:
            payload = {"inputs": prompt, "parameters": {"seed": seed}}
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            if response.status_code == 200:
                img_bytes = response.content
                dom_color = get_dominant_color(img_bytes)
                score = color_distance(dom_color, target_rgb)
                if score < best_score:
                    best_score = score
                    best_img = img_bytes

        if best_img:
            st.write(prompt)
            return best_img
        else:
            st.error("모든 이미지 생성에 실패했습니다.")
            return None

# Spotify에서 노래 검색 후 Deezer에서 미리 듣기 URL 가져오기
def search_songs(query):
    results = sp.search(q=query, limit=6, type='track')
    songs = []

    for track in results['tracks']['items']:
        song_name = track['name']
        artist_name = track['artists'][0]['name']

        # 검색 결과 중복 제외
        if any(s['name'] == song_name and s['artist'] == artist_name for s in songs):
            print(f"\n\nSkipping duplicate: {song_name} - {artist_name}")
            continue

        # Deezer에서 미리 듣기 URL 가져오기 (제목&아티스트 비교)
        deezer_data = get_deezer_preview_url(song_name, artist_name)
        preview_url = deezer_data if deezer_data else None

        # 앨범 이미지 가져오기 (없으면 빈 문자열)
        album_images = track['album'].get('images', [])
        image_url = album_images[0]['url'] if album_images else ""

        songs.append({
            "name": song_name,
            "artist": artist_name,
            "image": image_url,
            "deezer_preview_url": preview_url
        })
        
    return songs

st.title("🎵 Playlist imagenerator")
query = st.text_input("제목 혹은 가수를 입력하세요")

if st.button("검색") and query:
    if st.session_state.searched:
        selected_song_data = [s for s in st.session_state.songs if f"{s['name']} - {s['artist']}" in st.session_state.selected_songs]
        st.session_state.past_selected_songs.append(selected_song_data)
        st.session_state.songs = search_songs(query)
    else:
        st.session_state.songs = search_songs(query)
        st.session_state.searched = True

# 검색 결과가 변경될 때, 기존 선택 목록을 필터링
available_songs = [f"{s['name']} - {s['artist']}" for s in st.session_state.songs]
valid_selected_songs = [s for s in st.session_state.selected_songs if s in available_songs]  # ✅ 유효한 값만 유지

# 선택한 노래를 업데이트하는 함수
def update_selected_songs():
    selected = st.session_state.temp_selected_songs
    print(f"Selected songs: {selected}")

    if len(selected) > 5:
        st.toast("⚠️ 노래는 최대 5곡까지만 선택할 수 있어요!", icon="🚫")
        # 마지막 선택을 제거하여 5개로 제한
        st.session_state.temp_selected_songs = selected[:5]
        return

    st.session_state.selected_songs = selected

    # 과거 선택된 곡을 포함 (선택 결과 반영)
    flattened_past_songs = [song for sublist in st.session_state.past_selected_songs for song in sublist]
    st.session_state.selected_songs = flattened_past_songs + st.session_state.selected_songs


# 노래 선택 UI (유효한 값만 default로 설정)
st.multiselect(
    "노래를 선택하세요 (최대 5곡)",
    options=available_songs,
    default=valid_selected_songs[:5],  # 최대 5개까지만 유지
    key="temp_selected_songs",
    on_change=update_selected_songs
)

# past_selected_songs를 평탄화하고, selected_song_data와 결합
def get_selected_song_data():
    # 현재 선택된 곡 정보
    current_data = [
        s for s in st.session_state.songs
        if f"{s['name']} - {s['artist']}" in st.session_state.selected_songs
    ]

    # 평탄화 + 병합
    flattened_past_songs = [
        song for sublist in st.session_state.past_selected_songs for song in sublist
    ]
    combined = flattened_past_songs + current_data

    # name-artist 기준 중복 제거
    seen = set()
    unique = []
    for song in combined:
        key = f"{song['name']} - {song['artist']}"
        if key not in seen:
            seen.add(key)
            unique.append(song)

    return unique


# 선택한 노래를 가로 정렬로 표시
if st.session_state.selected_songs:
    st.write("### 선택한 노래")
    selected_song_data = get_selected_song_data()  
    
    # 선택된 노래들을 컬럼에 맞게 정렬하여 표시
    cols = st.columns(len(selected_song_data))
    for idx, song in enumerate(selected_song_data):
        with cols[idx]:
            st.image(song['image'], width=150)
            st.write(f"**{song['name']}**")
            st.write(song['artist'])

cols = st.columns(3)  # 열을 생성
style = cols[0].radio("**Illustrate Style**", ["Color", "Character", "Landscape", "Abstract"])  # 첫 번째 열에서 라디오 버튼
color = cols[1].color_picker("**Overall color**", "#ff0000")  # 두 번째 열에서 색상 선택기
seed_mode = cols[2].radio(
    "🎲 이미지 생성 방식",
    ["고정된 시드로 1장 생성", "여러 시드 중 최적 결과 선택"]
)

if st.session_state.selected_songs and st.button("표지 생성"):
    with st.spinner("플레이리스트 분석 중..."):
        selected_song_data = get_selected_song_data()  
        features_list = [extract_audio_features(s['deezer_preview_url']) for s in selected_song_data if s['deezer_preview_url']]
        valid_features = [f for f in features_list if f]
        aggregated_features = aggregate_features(valid_features) if valid_features else None
    if aggregated_features:
        with st.spinner("플레이리스트 표지 생성 중..."):
            image_url = generate_playlist_image(aggregated_features, style, color, seed_mode)
            if image_url:
                st.image(image_url, caption="생성된 플레이리스트 표지", width=250)
            else:
                st.toast("이미지 URL이 유효하지 않습니다.", icon="😢")
    else:
        for i in range(len(selected_song_data)):
            if not selected_song_data[i]['deezer_preview_url']:
                st.toast(f"{selected_song_data[i]['name']} - {selected_song_data[i]['artist']}의 미리듣기가 제공되지 않습니다.", icon="😢")
        st.error("오디오 분석을 위한 데이터가 충분하지 않습니다.")