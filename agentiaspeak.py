import streamlit as st
import requests
from deep_translator import GoogleTranslator
import whisperx
import torch
import os
import difflib
import io
import time
from pydub import AudioSegment

# --- ページ設定 ---
st.set_page_config(page_title="AGENTIA Speak Proβ", layout="wide")

# セッション状態の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "edit_list" not in st.session_state:
    st.session_state.edit_list = []

# --- デザイン ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #f8f9fa; }
    .char-pill {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        background: #e9ecef;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI解析エンジン (WhisperX) ---
@st.cache_resource
def get_whisperx_resources(lang_code):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # モデルサイズはStreamlit Cloudのメモリ制限を考慮し 'base' を推奨
    model = whisperx.load_model("base", device, compute_type="float32")
    return model, device

def align_audio_with_whisperx(audio_bytes, text, lang_code):
    model, device = get_whisperx_resources(lang_code)
    
    # --- 修正ポイント：BytesIOではなく物理ファイルを経由させる ---
    temp_audio_path = f"temp_align_{int(time.time())}.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)
    
    try:
        # whisperx.load_audio にファイルパス（文字列）を渡す
        audio_np = whisperx.load_audio(temp_audio_path)
        
        # 2. トランスクリプト
        result = model.transcribe(audio_np, batch_size=1, language=lang_code)
        
        # 3. アライメント実行
        model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_np, device, return_char_alignments=True)
        
        char_data = []
        for segment in result["segments"]:
            if "chars" in segment:
                for char_info in segment["chars"]:
                    if "start" in char_info:
                        char_data.append({
                            "char": char_info["char"],
                            "start": char_info["start"],
                            "end": char_info["end"]
                        })
        return char_data
    
    finally:
        # 使い終わったら確実に削除
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- メインロジック ---
col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("📜 テイク履歴 & AI解析")
    for item in st.session_state.history:
        with st.expander(f"{item['display_id']}: {item['text'][:15]}..."):
            st.audio(item["data"])
            # AIが解析した各文字のタイミングを表示
            chars_html = "".join([f"<span class='char-pill'>{c['char']}<br><small>{c['end']:.2f}</small></span>" for c in item["alignment"]])
            st.markdown(chars_html, unsafe_allow_html=True)

with col_left:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=350)
    
    # 生成セクション
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("🎙️ 音声生成")
        text_input = st.text_area("日本語テキストを入力", placeholder="例: こんにちは、本日のニュースをお伝えします。")
        lang_opt = st.selectbox("言語", ["日本語", "英語", "中国語", "スペイン語", "韓国語"])
        
        if st.button("テイクを生成", use_container_width=True):
            api_key = st.secrets.get("FISH_AUDIO_API_KEY")
            if text_input and api_key:
                with st.spinner("生成中..."):
                    l_map = {"日本語":"ja","英語":"en","中国語":"zh-CN","スペイン語":"es","韓国語":"ko"}
                    target_lang = l_map[lang_opt]
                    translated = GoogleTranslator(source='ja', target=target_lang).translate(text_input)
                    
                    res = requests.post("https://api.fish.audio/v1/tts",
                        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                        json={"text":translated, "format":"wav", "reference_id":"ffe7a84cf0e243359b28e6c3686bc9af"} # 例として男性ID
                    )
                    
                    if res.status_code == 200:
                        audio_bytes = res.content
                        # WhisperXでアライメント解析
                        alignment = align_audio_with_whisperx(audio_bytes, translated, target_lang)
                        
                        st.session_state.history.insert(0, {
                            "id": int(time.time()),
                            "display_id": f"T-{len(st.session_state.history)+1}",
                            "data": audio_bytes,
                            "text": translated,
                            "alignment": alignment
                        })
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 編集セクション
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("✂️ AI自動吸着編集")
        if not st.session_state.history:
            st.info("テイクを生成してください。")
        else:
            c1, c2 = st.columns(2)
            with c1:
                target_id = st.selectbox("テイク選択", [h["id"] for h in st.session_state.history], 
                                         format_func=lambda x: next(h["display_id"] for h in st.session_state.history if h["id"] == x))
                selected_take = next(h for h in st.session_state.history if h["id"] == target_id)
            
            with c2:
                # ここがAI吸着の肝：文字リストからカットポイントを選択
                char_options = [f"{i}: {c['char']} (〜{c['end']:.2f}s)" for i, c in enumerate(selected_take["alignment"])]
                selected_char_idx = st.selectbox("カットする文字を選択", range(len(char_options)), format_func=lambda x: char_options[x])
                
            if st.button("この文字の直後で切り出してリストに追加"):
                cutoff = selected_take["alignment"][selected_char_idx]["end"]
                # 簡易的に「開始0秒〜選択した文字の終了秒」までを追加
                st.session_state.edit_list.append({
                    "id": target_id,
                    "start": 0.0,
                    "end": cutoff,
                    "label": f"{selected_take['display_id']} の '{selected_take['alignment'][selected_char_idx]['char']}' まで"
                })

            if st.session_state.edit_list:
                st.markdown("---")
                for clip in st.session_state.edit_list:
                    st.text(f"✅ {clip['label']}")
                
                if st.button("AI結合実行 (フェード補正あり)", use_container_width=True):
                    # 前回の結合ロジック（pydub）を実行
                    final_wav = AudioSegment.empty()
                    for clip in st.session_state.edit_list:
                        source = next(h for h in st.session_state.history if h["id"] == clip["id"])
                        seg = AudioSegment.from_file(io.BytesIO(source["data"]))[clip["start"]*1000 : clip["end"]*1000]
                        final_wav += seg.fade_out(50).fade_in(50) # つなぎ目を50msでクロスフェード
                    
                    out_buf = io.BytesIO()
                    final_wav.export(out_buf, format="wav")
                    st.audio(out_buf.getvalue())
                    st.download_button("完成ファイルを保存", out_buf.getvalue(), "final.wav", "audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

st.caption("© 2026 Powered by FEIDIAS Inc.")