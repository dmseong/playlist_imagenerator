<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Spotify-1ED760?style=for-the-badge&logo=Spotify&logoColor=white"> <img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white">

<h1>🎵 Playlist Image Generator</h1>
    <p>This web application allows users to generate a playlist cover image based on the musical features of selected songs. By analyzing audio data, it creates a visual representation of the playlist's vibe.</p>

<h2>Features</h2>
    <ul>
        <li>Search songs by title or artist using the <span style="font-weight: bold; color: #1ED760">Spotify API</span>.</li>
        <li>Fetch preview URLs from <span style="font-weight: bold; color: #9933FF">Deezer API</span> for selected songs.</li>
        <li>Analyze audio features such as tempo, spectral centroid, and spectral bandwidth using <span style="font-weight: bold; color:rgba(233, 85, 164, 0.83)">Librosa</span>.</li>
        <li>Aggregate the features of selected songs to determine the playlist's mood.</li>
        <li>Generate a playlist cover using Hugging Face's <span style="font-weight: bold; color:rgba(105, 193, 255, 0.94)">Stable Diffusion</span> model based on the aggregated audio features.</li>
        <li>Customize the cover style (Color, Character, Landscape, or Abstract).</li>
    </ul>

<a href="https://playlistimagenerator.streamlit.app/" style="text-decoration: none; color: #FF4B4B" target="_blank">Playlist imagenerator</a>