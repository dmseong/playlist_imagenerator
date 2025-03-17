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
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-3.5-large"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Spotify API 설정
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_deezer_preview_url(song_name, artist_name):
    """Deezer API를 사용하여 특정 노래의 미리 듣기 URL을 가져옴"""
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

# 종합적인 분위기 계산
def aggregate_features(features_list):
    if not features_list:
        return None
    avg_features = {key: np.mean([f[key] for f in features_list]) for key in features_list[0]}
    st.write("### 플레이리스트 분석 결과")
    st.markdown(f"🎵 평균 템포: {avg_features['tempo']:.2f} BPM")
    st.markdown(f"""
                <p class="tooltip">🎶 평균 스펙트럴 센트로이드: {avg_features['spectral_centroid']:.2f}
                    <span class="tooltiptext">이 값이 높을수록 음색이 밝습니다.</span>
                </p>
                """, unsafe_allow_html=True)
    st.markdown(f"""
                <p class="tooltip">🎸 평균 스펙트럴 밴드위드: {avg_features['spectral_bandwidth']:.2f}
                    <span class="tooltiptext">이 값이 클수록 음악의 다이내믹 레인지가 넓습니다.</span>
                </p>
                """, unsafe_allow_html=True)
    return avg_features

# Stable Diffusion 이미지 생성
def generate_playlist_image(features, style, color):
    # 스타일 프롬프트
    style_prompt = {
        "Color": "Express the mood of the music using only gradients of different colors. You must not draw any objects.",
        "Chracter": "Please create a cover with a Japanese anime style character that matches the mood of the music.",
        "Landscape": "Create a cover that reflects the overall mood of the music in the form of a landscape.",
        "Abstract": "Create an abstract cover that captures the essence of the music."
    }.get(style, "Color")

    color_prompt = f" Use a gradient effect based on the {color}, and express the emotional mood of the music through this gradient."

    prompt = f"A playlist cover reflecting the overall musical vibe:"
    
    # 🎵 음악의 템포(속도) 분석
    if features['tempo'] > 160:
        prompt += " A very fast and high-energy track, often found in intense rock or electronic music."  # 매우 빠르고 에너제틱한 음악 (강렬한 록, 빠른 전자 음악)
    elif features['tempo'] > 130:
        prompt += " A fast and energetic rhythm, commonly heard in rock, punk, and dance music."  # 빠르고 에너지 넘치는 리듬 (록, 펑크, 댄스 음악)
    elif features['tempo'] > 100:
        prompt += " A moderately fast tempo, giving a vibrant and lively feel."  # 적당히 빠른 템포로 생동감 있는 분위기
    elif features['tempo'] > 70:
        prompt += " A balanced rhythm with a relaxed yet engaging pace."  # 균형 잡힌 리듬, 편안하면서도 몰입감 있는 템포
    else:
        prompt += " A slow and soothing track with a calm and peaceful atmosphere."  # 느리고 차분한 음악 (잔잔한 발라드, 어쿠스틱)

    # 🎸 음악의 음색(밝기) 분석 (스펙트럴 센트로이드)
    if features['spectral_centroid'] > 5500:
        prompt += " A bright and sharp sound, often associated with high-energy rock and metal."  # 밝고 날카로운 사운드 (에너지 넘치는 록, 메탈)
    elif features['spectral_centroid'] > 4000:
        prompt += " A slightly bright yet warm tone, commonly found in pop rock and alternative music."  # 다소 밝으면서도 따뜻한 톤 (팝 록, 얼터너티브)
    elif features['spectral_centroid'] > 2500:
        prompt += " A well-balanced sound with a mix of warmth and clarity."  # 따뜻함과 명료함이 조화된 사운드
    else:
        prompt += " A deep and mellow tone, often associated with acoustic and jazz music."  # 부드럽고 깊은 톤 (어쿠스틱, 재즈)

    # 🎶 음악의 사운드 다이내믹(폭) 분석 (스펙트럴 밴드위드)
    if features['spectral_bandwidth'] > 3000:
        prompt += " A highly dynamic and expressive sound with a wide frequency range."  # 매우 다이내믹하고 표현력이 강한 사운드
    elif features['spectral_bandwidth'] > 2000:
        prompt += " A vibrant and energetic texture, often found in rock and upbeat tracks."  # 생동감 있고 에너지 넘치는 질감 (록, 업템포 음악)
    elif features['spectral_bandwidth'] > 1200:
        prompt += " A smooth and clear sound with a mix of mellow and bright elements."  # 부드러우면서도 선명한 사운드 (밝은 요소와 부드러운 요소가 혼합)
    else:
        prompt += " A soft and warm sound with subtle variations, ideal for calm and acoustic music."  # 부드럽고 따뜻한 사운드, 차분한 음악에 적합 (어쿠스틱, 포크)
    
    prompt += f" {style_prompt} {color_prompt}"

    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        st.write(prompt)
        return response.content
    else:
        st.error("이미지 생성 실패!")
        return None

def search_songs(query):
    """Spotify에서 노래 검색 후 Deezer에서 미리 듣기 URL 가져오기"""
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

def update_selected_songs():
    """ 사용자가 선택한 곡을 session_state에 반영 """
    # temp_selected_songs를 session_state.selected_songs에 업데이트
    st.session_state.selected_songs = st.session_state.temp_selected_songs
    
    # past_selected_songs를 평탄화하여 selected_songs의 0번째에 삽입
    flattened_past_songs = [song for sublist in st.session_state.past_selected_songs for song in sublist]
    
    # past_selected_songs를 selected_songs 앞에 붙여줌
    st.session_state.selected_songs = flattened_past_songs + st.session_state.selected_songs


# 🔥 노래 선택 UI (유효한 값만 default로 설정)
st.multiselect(
    "노래를 선택하세요",
    options=available_songs,
    default=valid_selected_songs,  
    key="temp_selected_songs",
    on_change=update_selected_songs
)

# past_selected_songs를 평탄화하고, selected_song_data와 결합
def get_selected_song_data():
    selected_song_data = [s for s in st.session_state.songs if f"{s['name']} - {s['artist']}" in st.session_state.selected_songs]
    flattened_past_songs = [song for sublist in st.session_state.past_selected_songs for song in sublist]
    selected_song_data = flattened_past_songs + selected_song_data # past_selected_songs를 앞에 붙여서 결합
    return selected_song_data

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

cols = st.columns(2)  # 열을 생성
style = cols[0].radio("**Illustrate Style**", ["Color", "Character", "Landscape", "Abstract"])  # 첫 번째 열에서 라디오 버튼
color = cols[1].color_picker("**Overall color**", "#ff0000")  # 두 번째 열에서 색상 선택기

if st.session_state.selected_songs and st.button("표지 생성"):
    with st.spinner("플레이리스트 분석 중..."):
        selected_song_data = get_selected_song_data()  
        features_list = [extract_audio_features(s['deezer_preview_url']) for s in selected_song_data if s['deezer_preview_url']]
        valid_features = [f for f in features_list if f]
        aggregated_features = aggregate_features(valid_features) if valid_features else None
    if aggregated_features:
        with st.spinner("플레이리스트 표지 생성 중..."):
            image_url = generate_playlist_image(aggregated_features, style, color)
            if image_url:
                st.image(image_url, caption="생성된 플레이리스트 표지", width=250)
            else:
                st.toast("이미지 URL이 유효하지 않습니다.", icon="😢")
    else:
        st.error("오디오 분석을 위한 데이터가 충분하지 않습니다.")