"""
🎵 HitScore — Audio Intelligence Platform
Run with:  python3 -m streamlit run app.py
Install:   pip install librosa soundfile
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HitScore — Music AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f0eb !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #2a2420;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 15% 20%, rgba(255,182,142,0.25) 0%, transparent 55%),
        radial-gradient(ellipse 60% 60% at 85% 75%, rgba(180,210,255,0.20) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(200,240,200,0.10) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}

[data-testid="stMain"] { position: relative; z-index: 1; }
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

.block-container { padding: 2.5rem 3.5rem 5rem !important; max-width: 1300px !important; }

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 1rem;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #7a6a5a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(3rem, 6vw, 5.8rem);
    font-weight: 700;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #1a1410;
    margin-bottom: 1rem;
}
.hero-title em {
    font-style: italic;
    color: #c46a3a;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8a7a70;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 0;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin: 2.5rem 0;
    flex-wrap: wrap;
}
.stat-chip {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    backdrop-filter: blur(20px);
    min-width: 110px;
}
.stat-chip-val {
    font-family: 'Fraunces', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: #1a1410;
    line-height: 1;
    display: block;
}
.stat-chip-label {
    font-size: 0.65rem;
    color: #a09080;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
    display: block;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.60);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 24px;
    padding: 1.8rem;
    backdrop-filter: blur(30px);
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.9), transparent);
}
.card-title {
    font-family: 'Fraunces', serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #a09080;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,0,0,0.06);
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.5) !important;
    border: 2px dashed rgba(196,106,58,0.3) !important;
    border-radius: 18px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(196,106,58,0.6) !important;
    background: rgba(255,255,255,0.7) !important;
}
[data-testid="stFileUploader"] label {
    color: #7a6a5a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Score display ── */
.score-main {
    text-align: center;
    padding: 2rem 1rem;
}
.score-big {
    font-family: 'Fraunces', serif;
    font-size: 7rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.04em;
    display: block;
}
.score-sub {
    font-size: 0.85rem;
    color: #a09080;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.score-badge {
    display: inline-block;
    margin-top: 1.2rem;
    padding: 0.45rem 1.4rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Progress ── */
.prog-outer {
    background: rgba(0,0,0,0.06);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
    margin: 1.5rem 0 0.5rem;
}
.prog-inner {
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
}
.prog-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 0.62rem;
    color: #b0a090;
    letter-spacing: 0.05em;
}

/* ── Feature bars ── */
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.75rem;
}
.feat-name {
    font-size: 0.72rem;
    color: #7a6a5a;
    width: 80px;
    flex-shrink: 0;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.feat-track {
    flex: 1;
    height: 6px;
    background: rgba(0,0,0,0.06);
    border-radius: 50px;
    overflow: hidden;
}
.feat-fill {
    height: 100%;
    border-radius: 50px;
}
.feat-val {
    font-size: 0.7rem;
    color: #a09080;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Advice cards ── */
.advice-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 0.5rem;
}
.advice-chip {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    flex: 1;
    min-width: 180px;
}
.advice-icon { font-size: 1.3rem; margin-bottom: 0.4rem; display: block; }
.advice-title { font-size: 0.72rem; font-weight: 600; color: #4a3a2a; margin-bottom: 0.2rem; }
.advice-body { font-size: 0.68rem; color: #8a7a6a; line-height: 1.5; }

/* ── Slider overrides ── */
[data-testid="stSlider"] label {
    color: #6a5a4a !important;
    font-size: 0.78rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #c46a3a !important;
    border: 2px solid #f5f0eb !important;
    box-shadow: 0 2px 8px rgba(196,106,58,0.35) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: rgba(196,106,58,0.15) !important;
}

/* Tab override */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    color: #8a7a6a !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #c46a3a !important;
    border-bottom-color: #c46a3a !important;
}

[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/features.json') as f:
        features = json.load(f)
    summary = {}
    if os.path.exists('models/summary.json'):
        with open('models/summary.json') as f:
            summary = json.load(f)
    return model, scaler, features, summary

try:
    model, scaler, FEATURES, summary = load_artifacts()
except Exception as e:
    st.error(f"Model not found: {e} — please run the notebook first.")
    st.stop()


# ── Librosa extraction ────────────────────────────────────────────────────────
def extract_features_librosa(audio_path):
    try:
        import librosa
        y, sr = librosa.load(audio_path, duration=60, mono=True)

        tempo, beats   = librosa.beat.beat_track(y=y, sr=sr)
        rms            = float(np.mean(librosa.feature.rms(y=y)))
        energy         = float(np.clip(rms * 40, 0, 1))
        loudness       = float(np.clip(20 * np.log10(rms + 1e-9), -60, 0))
        spectral_cent  = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr            = librosa.feature.zero_crossing_rate(y)
        mfcc           = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma         = librosa.feature.chroma_stft(y=y, sr=sr)
        harmonic, perc = librosa.effects.hpss(y)
        harm_energy    = float(np.mean(librosa.feature.rms(y=harmonic)))
        perc_energy    = float(np.mean(librosa.feature.rms(y=perc)))

        acousticness  = float(np.clip(1.0 - energy, 0, 1))
        danceability  = float(np.clip(perc_energy * 60, 0, 1))
        valence       = float(np.clip(np.mean(chroma) * 5, 0, 1))
        speechiness   = float(np.clip(np.mean(zcr) * 3, 0, 1))
        liveness      = float(np.clip(np.std(mfcc[0]) / 50, 0, 1))
        instrumentalness = float(np.clip(1 - speechiness * 2, 0, 1))

        # Bass energy (low freq)
        stft   = np.abs(librosa.stft(y))
        freqs  = librosa.fft_frequencies(sr=sr)
        bass_idx = np.where(freqs < 250)[0]
        bass_energy = float(np.mean(stft[bass_idx, :]))
        bass_norm = float(np.clip(bass_energy / (np.mean(stft) + 1e-9) / 3, 0, 1))

        return {
            'energy': energy, 'loudness': loudness,
            'danceability': danceability, 'valence': valence,
            'acousticness': acousticness, 'speechiness': speechiness,
            'liveness': liveness, 'instrumentalness': instrumentalness,
            'tempo': float(tempo), 'bass': bass_norm,
            'spectral_centroid': float(np.mean(spectral_cent)),
            'key': int(np.argmax(np.mean(chroma, axis=1))),
            'mode': 1,
        }, None
    except ImportError:
        return None, "librosa not installed — run: pip install librosa soundfile"
    except Exception as e:
        return None, str(e)


# ── Build feature vector ──────────────────────────────────────────────────────
def build_features(energy, valence, danceability, loudness,
                   acousticness, speechiness, tempo, liveness):
    row = {
        'energy': energy, 'valence': valence, 'danceability': danceability,
        'loudness': loudness, 'acousticness': acousticness, 'tempo': tempo,
        'speechiness': speechiness, 'liveness': liveness,
        'energy_valence': energy * valence,
        'dance_energy': danceability * energy,
        'acoustic_energy_gap': energy - acousticness,
        'loudness_norm': (loudness + 60) / 60,
        'speech_ratio': speechiness / (energy + 1e-6),
        # New/unknown-artist tracks (uploads, manual entry) fall back to the
        # training set's global average popularity rather than a guessed
        # constant — see 'global_mean_popularity' in models/summary.json.
        'artist_mean_pop': summary.get('global_mean_popularity', 45),
        'artist_song_count': 0,
    }
    return pd.DataFrame([row])[FEATURES]


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-pill">🎵 Music Intelligence Platform</div>
    <div class="hero-title">Will your track<br><em>hit different?</em></div>
    <div class="hero-sub">Upload an MP3 and our model auto-reads your song's DNA — or tune sliders manually. Get a Spotify popularity prediction with SHAP-powered advice.</div>
</div>
""", unsafe_allow_html=True)

# ── Ambient background sound toggle ────────────────────────────────────────
# A soft, generative ambient pad (built live with the Web Audio API, not a
# licensed track) that plays quietly behind the app. Starts on click only —
# browsers block audio autoplay without a user gesture anyway, and this way
# nobody gets surprised by sound on page load.
components.html("""
<div id="music-toggle" class="music-toggle" onclick="toggleAmbientMusic()">
    <span class="music-dot" id="music-dot"></span>
    <span id="music-label">Ambient sound</span>
</div>
<style>
    html, body { background: transparent !important; margin: 0; padding: 0; }
    .music-toggle {
        display: inline-flex; align-items: center; gap: 0.55rem;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 50px;
        padding: 0.5rem 1.1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.78rem; font-weight: 600; color: #7a6a5a;
        cursor: pointer; transition: all 0.2s ease; user-select: none;
        width: fit-content;
    }
    .music-toggle:hover { background: rgba(255,255,255,1); }
    .music-dot { width: 8px; height: 8px; border-radius: 50%; background: #c9bfae; transition: background 0.2s ease; }
    .music-toggle.playing .music-dot { background: #ff9a62; animation: pulse-dot 1.3s ease-in-out infinite; }
    @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
</style>
<script>
    let audioCtx = null;
    let musicNodes = [];
    let isPlaying = false;

    function toggleAmbientMusic() {
        const toggle = document.getElementById('music-toggle');
        const label  = document.getElementById('music-label');

        if (!isPlaying) {
            if (!audioCtx) { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
            if (audioCtx.state === 'suspended') { audioCtx.resume(); }

            const masterGain = audioCtx.createGain();
            masterGain.gain.value = 0;
            masterGain.connect(audioCtx.destination);
            masterGain.gain.linearRampToValueAtTime(0.045, audioCtx.currentTime + 1.5);

            // Soft C-major pad, four slowly-detuned sine tones
            const freqs = [130.81, 164.81, 196.00, 261.63];
            freqs.forEach((f, i) => {
                const osc = audioCtx.createOscillator();
                osc.type = 'sine';
                osc.frequency.value = f;

                const lfo = audioCtx.createOscillator();
                lfo.frequency.value = 0.07 + i * 0.02;
                const lfoGain = audioCtx.createGain();
                lfoGain.gain.value = 2.5;
                lfo.connect(lfoGain);
                lfoGain.connect(osc.frequency);

                const noteGain = audioCtx.createGain();
                noteGain.gain.value = 0.25 / freqs.length;
                osc.connect(noteGain);
                noteGain.connect(masterGain);

                osc.start(); lfo.start();
                musicNodes.push(osc, lfo);
            });
            musicNodes.push(masterGain);

            isPlaying = true;
            toggle.classList.add('playing');
            label.innerText = 'Ambient sound on';
        } else {
            const masterGain = musicNodes[musicNodes.length - 1];
            masterGain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.8);
            setTimeout(() => {
                musicNodes.forEach(n => { try { n.stop && n.stop(); } catch (e) {} });
                musicNodes = [];
            }, 900);
            isPlaying = false;
            toggle.classList.remove('playing');
            label.innerText = 'Ambient sound';
        }
    }
</script>
""", height=60)

# Stats
r2   = summary.get('test_r2', 0.81)
mae  = summary.get('test_mae', 4.93)
rmse = summary.get('test_rmse', 9.56)
nf   = summary.get('n_features', 15)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-chip"><span class="stat-chip-val">{r2}</span><span class="stat-chip-label">R² Score</span></div>
    <div class="stat-chip"><span class="stat-chip-val">±{mae}</span><span class="stat-chip-label">Mean Error</span></div>
    <div class="stat-chip"><span class="stat-chip-val">{rmse}</span><span class="stat-chip-label">RMSE</span></div>
    <div class="stat-chip"><span class="stat-chip-val">{nf}</span><span class="stat-chip-label">Features</span></div>
    <div class="stat-chip"><span class="stat-chip-val">114k</span><span class="stat-chip-label">Tracks Trained</span></div>
    <div class="stat-chip"><span class="stat-chip-val">XGB</span><span class="stat-chip-label">Model</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ── INPUT: Tabs for upload vs manual ────────────────────────────────────────
tab1, tab2 = st.tabs(["🎵 Upload your MP3  ", "🎛️ Manual sliders  "])

extracted = None

with tab1:
    st.markdown('<br>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your MP3, WAV, or FLAC here — we'll auto-extract all audio features",
        type=['mp3','wav','flac','ogg','m4a'],
        label_visibility='visible'
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.'+uploaded.name.split('.')[-1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("🔬 Analysing your track — reading bass, tempo, energy..."):
            feats, err = extract_features_librosa(tmp_path)

        if err:
            st.warning(f"⚠️ {err}")
            st.info("Install librosa: `pip install librosa soundfile` then restart the app.")
        else:
            extracted = feats
            st.success(f"✅ **{uploaded.name}** analysed! Tempo detected: **{feats['tempo']:.0f} BPM** · Key: **{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][feats['key']]}**")

            # Show extracted bars
            st.markdown('<br><div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Extracted Audio Features</div>', unsafe_allow_html=True)
            feat_display = {
                '⚡ Energy': feats['energy'], '💃 Danceability': feats['danceability'],
                '😊 Valence': feats['valence'], '🎸 Acousticness': feats['acousticness'],
                '🎤 Speechiness': feats['speechiness'], '🎪 Liveness': feats['liveness'],
                '🎹 Instrumental': feats['instrumentalness'], '🎸 Bass': feats['bass'],
            }
            pastel_colors = ['#ffb5a7','#ffd6a5','#fdffb6','#caffbf','#9bf6ff','#a0c4ff','#bdb2ff','#ffc6ff']
            bar_html = ''
            for (name, val), color in zip(feat_display.items(), pastel_colors):
                pct = int(val * 100)
                bar_html += f'''
                <div class="feat-row">
                    <div class="feat-name">{name}</div>
                    <div class="feat-track"><div class="feat-fill" style="width:{pct}%;background:{color};"></div></div>
                    <div class="feat-val">{val:.2f}</div>
                </div>'''
            st.markdown(bar_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<br>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        m_energy       = st.slider("⚡ Energy",       0.0, 1.0, 0.65, 0.01)
        m_danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.70, 0.01)
    with sc2:
        m_valence      = st.slider("😊 Valence",      0.0, 1.0, 0.55, 0.01)
        m_acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.15, 0.01)
    with sc3:
        m_loudness     = st.slider("🔊 Loudness (dB)",-60.0, 0.0, -6.0, 0.5)
        m_speechiness  = st.slider("🎤 Speechiness",  0.0, 1.0, 0.05, 0.01)
    with sc4:
        m_tempo        = st.slider("🥁 Tempo (BPM)",  50.0, 220.0, 120.0, 1.0)
        m_liveness     = st.slider("🎪 Liveness",      0.0, 1.0, 0.15, 0.01)


# ── Resolve final feature values ──────────────────────────────────────────────
if extracted:
    energy       = extracted['energy']
    valence      = extracted['valence']
    danceability = extracted['danceability']
    loudness     = extracted['loudness']
    acousticness = extracted['acousticness']
    speechiness  = extracted['speechiness']
    tempo        = extracted['tempo']
    liveness     = extracted['liveness']
else:
    energy       = m_energy
    valence      = m_valence
    danceability = m_danceability
    loudness     = m_loudness
    acousticness = m_acousticness
    speechiness  = m_speechiness
    tempo        = m_tempo
    liveness     = m_liveness


# ── Predict ───────────────────────────────────────────────────────────────────
input_df     = build_features(energy, valence, danceability, loudness,
                              acousticness, speechiness, tempo, liveness)
input_scaled = scaler.transform(input_df)
score        = float(np.clip(model.predict(input_scaled)[0], 0, 100))

if score >= 75:
    tag, color, bg, emoji = "Certified Hit", "#2d6a4f", "#d8f3dc", "🔥"
elif score >= 58:
    tag, color, bg, emoji = "Chart Potential", "#7a4f00", "#fff3cd", "📈"
elif score >= 40:
    tag, color, bg, emoji = "Solid Track", "#1a3a6a", "#dbeafe", "🎵"
else:
    tag, color, bg, emoji = "Needs Work", "#7a1a1a", "#fee2e2", "📉"


# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
r1, r2c, r3 = st.columns([1.1, 1, 1.4])

# Score card
with r1:
    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <div class="card-title">Popularity Score</div>
        <div class="score-main">
            <span class="score-big" style="color:{color};">{score:.0f}</span>
            <div class="score-sub">out of 100</div>
            <div class="score-badge" style="background:{bg}; color:{color}; border:1px solid {color}30;">
                {emoji} {tag}
            </div>
        </div>
        <div class="prog-outer">
            <div class="prog-inner" style="width:{score}%; background:linear-gradient(90deg,{color}80,{color});"></div>
        </div>
        <div class="prog-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
        <div style="font-size:0.65rem; color:#c0b0a0; margin-top:1.5rem; letter-spacing:0.1em;">
            XGBoost · Optuna · SHAP · 114k tracks
        </div>
    </div>
    """, unsafe_allow_html=True)

# Radar
with r2c:
    radar_labels = ['Energy', 'Dance', 'Valence', 'Acoustic', 'Speech', 'Live']
    radar_vals   = [energy, danceability, valence, acousticness, speechiness, liveness]
    N = len(radar_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_r = radar_vals + radar_vals[:1]

    pastel_fills = ['#ffb5a7','#ffd6a5','#fdffb6','#caffbf','#9bf6ff','#a0c4ff']

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff00')
    ax.set_facecolor('#ffffff00')

    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r]*len(angles), color='#2a2420', alpha=0.06, linewidth=0.8)
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1], color='#2a2420', alpha=0.06, linewidth=0.8)

    ax.fill(angles, vals_r, color=color, alpha=0.12)
    ax.plot(angles, vals_r, color=color, linewidth=2.5)
    for a, v, c in zip(angles[:-1], radar_vals, pastel_fills):
        ax.scatter(a, v, color=color, s=45, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, color='#7a6a5a', size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['polar'].set_color('#2a242010')

    plt.tight_layout()
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# SHAP + advice
with r3:
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_scaled)
        impacts   = pd.Series(shap_vals[0], index=FEATURES).sort_values(key=abs, ascending=False).head(8)

        fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
        fig2.patch.set_facecolor('#ffffff00')
        ax2.set_facecolor('#ffffff00')

        bar_colors = ['#caffbf' if v > 0 else '#ffb5a7' for v in impacts.values[::-1]]
        b2 = ax2.barh(
            [f.replace('_',' ') for f in impacts.index[::-1]],
            impacts.values[::-1],
            color=bar_colors, height=0.55, edgecolor='none'
        )
        for bar, val in zip(b2, impacts.values[::-1]):
            ax2.text(val + (0.05 if val >= 0 else -0.05),
                     bar.get_y() + bar.get_height()/2,
                     f'{val:+.1f}', va='center',
                     ha='left' if val >= 0 else 'right',
                     color='#7a6a5a', fontsize=7.5)

        ax2.axvline(0, color='#2a242020', linewidth=1)
        ax2.set_xlabel('Impact on score', color='#a09080', fontsize=8)
        ax2.tick_params(colors='#7a6a5a', labelsize=8)
        for spine in ax2.spines.values(): spine.set_visible(False)
        ax2.set_title('Why this score?', color='#2a2420', fontsize=10,
                      fontweight='bold', pad=10, loc='left',
                      fontfamily='serif')
        plt.tight_layout()
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP unavailable: {e}")


# ── Actionable advice ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🎯 How to Improve Your Track</div>', unsafe_allow_html=True)

advice = []
if danceability < 0.6:
    advice.append(("💃", "Boost Danceability", f"Currently {danceability:.2f}. Add a stronger, more consistent beat — aim for 0.70+. Tighten your kick/snare pattern."))
if energy < 0.55:
    advice.append(("⚡", "Raise Energy", f"Currently {energy:.2f}. Layer more instruments, increase compression, or add a drop section. Aim for 0.65+."))
if loudness < -12:
    advice.append(("🔊", "Louder Master", f"Currently {loudness:.1f} dB. Most Spotify hits sit between -8 and -4 dB LUFS. Consider a louder master."))
if valence < 0.4:
    advice.append(("😊", "Add Positivity", f"Low valence ({valence:.2f}) means your track sounds darker/sadder. A brighter chord progression can push this up."))
if acousticness > 0.7 and energy < 0.5:
    advice.append(("🎸", "Electric Elements", f"High acousticness ({acousticness:.2f}) + low energy. Adding electric instruments or production could broaden appeal."))
if tempo < 90 or tempo > 160:
    advice.append(("🥁", "Check Your BPM", f"Tempo is {tempo:.0f} BPM. Most chart hits fall between 90–140 BPM. Consider re-arranging if outside this range."))
if not advice:
    advice.append(("✨", "Already Optimised", "Your track's audio features are well-balanced for chart performance. Focus on promotion and release strategy!"))
    advice.append(("🎯", "Artist Popularity", "The biggest predictor is artist_mean_pop — building your catalog and fanbase matters more than any single audio feature."))

chips = ''
for icon, title, body in advice[:4]:
    chips += f'''
    <div class="advice-chip">
        <span class="advice-icon">{icon}</span>
        <div class="advice-title">{title}</div>
        <div class="advice-body">{body}</div>
    </div>'''

st.markdown(f'<div class="advice-row">{chips}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── Derived signals ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🔬 Computed Audio Signals</div>', unsafe_allow_html=True)

d_cols = st.columns(5)
derived = {
    "Upbeat Energy": (round(energy * valence, 3), "E × V", "#ffd6a5"),
    "Club Factor":   (round(danceability * energy, 3), "D × E", "#caffbf"),
    "Electric Gap":  (round(energy - acousticness, 3), "E − A", "#9bf6ff"),
    "Loud Norm":     (round((loudness + 60) / 60, 3), "(L+60)/60", "#ffc6ff"),
    "Speech Ratio":  (round(speechiness / (energy + 1e-6), 3), "S / E", "#ffb5a7"),
}
for col, (label, (val, formula, chip_color)) in zip(d_cols, derived.items()):
    with col:
        st.markdown(f"""
        <div style="background:{chip_color}30; border:1px solid {chip_color}80;
                    border-radius:16px; padding:1.1rem; text-align:center;">
            <div style="font-size:0.6rem; color:#8a7a6a; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:0.5rem;">{label}</div>
            <div style="font-family:'Fraunces',serif; font-size:1.9rem;
                        font-weight:600; color:#2a2420; line-height:1;">{val}</div>
            <div style="font-size:0.62rem; color:#a09080; margin-top:0.3rem;
                        font-family:monospace;">{formula}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:4rem; padding-top:2rem;
            border-top:1px solid rgba(0,0,0,0.06);">
    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.65rem;
                letter-spacing:0.25em; color:#c0b0a0; text-transform:uppercase;">
        XGBoost · Optuna Hypertuning · SHAP Explainability · Librosa Audio Analysis · 114k Tracks
    </div>
</div>
""", unsafe_allow_html=True)
