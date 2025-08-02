<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Spotify-1ED760?style=for-the-badge&logo=Spotify&logoColor=white"> <img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white"> <img src="https://img.shields.io/badge/Deezer-FEAA2D?style=for-the-badge&logo=deezer&logoColor=white">

# 🎵 Playlist Image Generator
This web application allows users to generate a playlist cover image based on the musical features of selected songs. By analyzing audio data, it creates a visual representation of the playlist's vibe.

**Development Period**: 2025.03

## Features
- Search songs by title or artist using the **[Spotify API](https://developer.spotify.com/)**.
- Fetch preview URLs from **[Deezer API](https://developers.deezer.com/)** for selected songs.
- Analyze audio features such as tempo, spectral centroid, and spectral bandwidth using **Librosa**.
- Aggregate the features of selected songs to determine the playlist's mood.
- Generate a playlist cover using Hugging Face's **[Stable Diffusion xl](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)** model based on the aggregated audio features.
- Customize the cover style (Color, Character, Landscape, or Abstract).
---------------
<img width="1920" height="1080" alt="2312446_김성희_발표자료01" src="https://github.com/user-attachments/assets/4d6bea2a-6be7-4d12-b892-1a915dc69fb5" />
<img width="1920" height="1080" alt="2312446_김성희_발표자료02" src="https://github.com/user-attachments/assets/257fafb5-3014-42e5-bc98-397e3f190f36" />
<img width="1920" height="1080" alt="2312446_김성희_발표자료03" src="https://github.com/user-attachments/assets/7358eb19-0e00-4a9a-a1ed-3b172a377151" />
<img width="1920" height="1080" alt="2312446_김성희_발표자료04" src="https://github.com/user-attachments/assets/df75da49-0355-4fa2-996f-20b7af1fe6c3" />
<img width="1920" height="1080" alt="2312446_김성희_발표자료05" src="https://github.com/user-attachments/assets/191cdff7-0529-4c27-b489-38be097719fd" />

## 🚀 Live Demo
**[Playlist imagenerator](https://playlistimagenerator.streamlit.app/)**
