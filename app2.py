# app.py
# Streamlit frontend — improved, modular, robust version
# Features:
# - Robust CSV/artifact auto-load (from repo or uploads)
# - Safe TensorFlow loading only when available
# - Caching for heavy operations
# - Clean EDA, content-based fallback + CNN-driven recommendations
# - Anomaly detection (Autoencoder or IsolationForest)
# - Downloadable recommendation results
# - Helpful error messages and progress feedback

import os
import io
import json
import pickle
import logging
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

# Optional TensorFlow imports (lazy)
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rec_app")

st.set_page_config(page_title="Recommendation System", page_icon="🧠", layout="wide")

# ---- Styles ----
st.markdown(
    """
    <style>
    h1, h2, h3 {color:#0b3d91}
    .section-card {background:#fff;border:1px solid #e7ecf3;border-radius:12px;padding:16px;margin-bottom:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Constants ----
REQUIRED_EVENTS_COLS = {"timestamp", "visitorid", "event", "itemid"}
REQUIRED_PROPS_COLS = {"timestamp", "itemid", "property", "value"}
REQUIRED_CAT_COLS = {"categoryid", "parentid"}

# ---- Helpers ----

def clean_numeric_str(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    toks = []
    for t in s.split():
        t = t.replace(",", ".")
        if t.startswith("n"):
            t = t[1:]
        try:
            float(t)
            toks.append(t)
        except Exception:
            continue
    if not toks:
        return np.nan
    try:
        return float(toks[-1])
    except Exception:
        return np.nan


def ensure_datetime_ms(series: pd.Series) -> pd.Series:
    def _to_dt(x):
        if pd.isna(x):
            return pd.NaT
        try:
            xv = float(x)
            if xv > 1e12:
                return pd.to_datetime(int(xv), unit="ms", errors="coerce")
            elif xv > 1e10:
                return pd.to_datetime(int(xv), unit="us", errors="coerce")
            else:
                return pd.to_datetime(int(xv), unit="s", errors="coerce")
        except Exception:
            return pd.to_datetime(x, errors="coerce")

    return series.apply(_to_dt)


@st.cache_data(show_spinner=False)
def load_csv(path: str, uploaded_file, expected_cols: Optional[set] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return None, f"Missing: {path}"
    except Exception as e:
        logger.exception("Failed to read CSV: %s", path)
        return None, f"Error reading {path}: {e}"

    df.columns = [c.strip().lower() for c in df.columns]
    if expected_cols:
        missing = set([c.lower() for c in expected_cols]) - set(df.columns)
        if missing:
            return None, f"Missing columns in {path or 'uploaded file'}: {missing}"
    return df, None


def latest_item_property(item_props: pd.DataFrame, prop_name: str) -> Dict[str, str]:
    sub = item_props[item_props["property"] == prop_name].sort_values("timestamp")
    return sub.groupby("itemid")["value"].last().to_dict()


def build_user_features(events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in events_df.groupby("visitorid"):
        total = len(g)
        views = (g["event"] == "view").sum()
        adds = (g["event"] == "addtocart").sum()
        buys = (g["event"] == "transaction").sum()
        rows.append({
            "visitorid": uid,
            "total_events": total,
            "views": views,
            "adds": adds,
            "buys": buys,
            "add_rate": adds / total if total else 0.0,
            "conv_rate": buys / total if total else 0.0,
        })
    if not rows:
        return pd.DataFrame(columns=["visitorid", "total_events", "views", "adds", "buys", "add_rate", "conv_rate"]).set_index("visitorid")
    return pd.DataFrame(rows).set_index("visitorid")


@st.cache_data
def prepare_content_matrix(item_props: pd.DataFrame, drop_props: Optional[set] = None):
    if drop_props is None:
        drop_props = {"available", "categoryid"}
    pp = (
        item_props[~item_props["property"].isin(drop_props)]
        .sort_values("timestamp")
        .groupby(["itemid", "property"]) ["value"]
        .last()
        .unstack(fill_value="")
    )
    pp = pp.fillna("")
    pp["__text__"] = pp.astype(str).apply(lambda r: " ".join(r.values.tolist()), axis=1)
    return pp


@st.cache_data
def fit_tfidf(texts: List[str], max_features: int = 5000):
    vect = TfidfVectorizer(max_features=max_features)
    mat = vect.fit_transform(texts)
    return vect, mat


def recommend_content_based(events: pd.DataFrame, item_props: pd.DataFrame, visitorid: str, topn: int = 10, exclude_viewed: bool = True):
    pp = prepare_content_matrix(item_props)
    vect, item_mat = fit_tfidf(pp["__text__"].values.tolist())

    uviews = events[(events["visitorid"] == visitorid) & (events["event"] == "view")]
    viewed_ids = [str(i) for i in uviews["itemid"].tolist()]

    if len(viewed_ids) == 0:
        pop = events[events["event"] == "view"]["itemid"].value_counts().index[:topn].astype(str).tolist()
        return pop, "Cold-start: showing popular items"

    sub = pp.loc[pp.index.astype(str).isin(viewed_ids)]
    if sub.empty:
        pop = events[events["event"] == "view"]["itemid"].value_counts().index[:topn].astype(str).tolist()
        return pop, "No property overlap: showing popular items"

    sub_tfidf = vect.transform(sub["__text__"].values)
    user_vec = sub_tfidf.mean(axis=0)
    sims = cosine_similarity(user_vec, item_mat).ravel()
    pp = pp.assign(similarity=sims)
    if exclude_viewed:
        pp = pp.loc[~pp.index.astype(str).isin(viewed_ids)]
    recs = pp.sort_values("similarity", ascending=False).index.astype(str).tolist()[:topn]
    return recs, "Content-based recommendations"


# ---- UI: Sidebar inputs ----
st.sidebar.header("Data & Artifacts")
use_cleaned = st.sidebar.checkbox("Use cleaned CSV filenames (from repo)", value=True)
up_events = st.sidebar.file_uploader("events.csv (or cleaned)", type=["csv"])
up_cat = st.sidebar.file_uploader("category_tree.csv (or cleaned)", type=["csv"])
up_props = st.sidebar.file_uploader("item_properties.csv (or cleaned)", type=["csv"])

# default paths from repo
default_events = "events_cleaned.csv" if use_cleaned else "events.csv"
default_cat = "category_tree_cleaned.csv" if use_cleaned else "category_tree.csv"
default_props = "item_properties_cleaned_n.csv" if use_cleaned else "item_properties.csv"

events, e_err = load_csv(default_events, up_events, REQUIRED_EVENTS_COLS)
category_tree, c_err = load_csv(default_cat, up_cat, REQUIRED_CAT_COLS)
item_props, p_err = load_csv(default_props, up_props, REQUIRED_PROPS_COLS)

if any([e_err, c_err, p_err]):
    st.error(f"Data loading issues:\n- {e_err}\n- {c_err}\n- {p_err}")
    st.stop()

# ---- Basic cleaning ----
events["timestamp"] = ensure_datetime_ms(events["timestamp"])  
item_props["timestamp"] = ensure_datetime_ms(item_props["timestamp"])  

events = events.dropna(subset=["timestamp", "visitorid", "event", "itemid"]).copy()
item_props = item_props.dropna(subset=["timestamp", "itemid", "property", "value"]).copy()
item_props["value"] = item_props["value"].apply(clean_numeric_str).fillna(item_props["value"])  

st.title("🧠 Recommendation System — Improved")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📈 EDA", "🛒 Recommendations", "🚨 Anomaly Detection"])

with tab1:
    st.subheader("Data summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Events", f"{len(events):,}")
    c2.metric("Item properties", f"{len(item_props):,}")
    c3.metric("Categories", f"{len(category_tree):,}")

    with st.expander("Event distribution and trends", expanded=True):
        ev_counts = events["event"].value_counts()
        st.plotly_chart(px.bar(ev_counts, title="Events by type"), use_container_width=True)

        ed = events.copy()
        ed["date"] = ed["timestamp"].dt.date
        daily = ed[ed["event"] == "view"].groupby("date").size().reset_index(name="views")
        if not daily.empty:
            st.plotly_chart(px.line(daily, x="date", y="views", title="Daily views"), use_container_width=True)
        else:
            st.info("No view events found.")

    with st.expander("Top properties", expanded=False):
        noisy = {"available", "categoryid"}
        top_props = item_props[~item_props["property"].isin(noisy)]["property"].value_counts().head(30)
        st.plotly_chart(px.bar(top_props, title="Top properties"), use_container_width=True)

with tab2:
    st.subheader("Property-based recommendation (CNN if available, else content-based fallback)")

    # target property
    target_prop = st.selectbox("Choose target property (used for tokenization)", item_props["property"].value_counts().index.tolist())

    # artifact paths
    model_path = st.text_input("CNN model path (h5)", value="cnn_model.h5")
    tokenizer_path = st.text_input("Tokenizer JSON (tokenizer.json)", value="tokenizer.json")
    labelenc_path = st.text_input("LabelEncoder (pickle)", value="labelencoder.pkl")

    cnn_model = None
    tokenizer = None
    label_enc = None

    if TF_AVAILABLE and os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(labelenc_path):
        try:
            cnn_model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                tok_json = json.load(f)
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tok_json)
            with open(labelenc_path, "rb") as f:
                label_enc = pickle.load(f)
            st.success("Loaded CNN recommender artifacts")
        except Exception as e:
            st.warning(f"Could not load TF artifacts: {e}")
            logger.exception("TF load failed")
    else:
        st.info("CNN artifacts not available or TF missing — using content-based fallback")

    visitor_list = events["visitorid"].astype(str).unique().tolist()
    chosen_user = st.selectbox("Pick a visitor", visitor_list[:5000])
    topN = st.slider("Number of recommendations", min_value=5, max_value=50, value=10)

    if st.button("Recommend"):
        with st.spinner("Generating recommendations..."):
            if cnn_model is not None and tokenizer is not None and label_enc is not None:
                # build tokens from user's prior views
                item_latest = latest_item_property(item_props, target_prop)
                hist = events[(events["visitorid"] == chosen_user) & (events["event"] == "view")].sort_values("timestamp")
                tokens = [str(item_latest.get(i, "")) for i in hist["itemid"].values if str(item_latest.get(i, "")).strip()]

                if not tokens:
                    recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                    st.info("No tokens found from history — falling back to content-based")
                    st.write(recs)
                else:
                    text = " ".join(tokens)
                    seq = tokenizer.texts_to_sequences([text])
                    pad = pad_sequences(seq, maxlen=50, padding="post")
                    probs = cnn_model.predict(pad, verbose=0)[0]
                    top_idx = np.argsort(probs)[::-1][:5]
                    preds = label_enc.inverse_transform(top_idx)
                    st.write("Predicted target property values:", preds.tolist())

                    # find candidate items
                    item_latest_map = latest_item_property(item_props, target_prop)
                    candidates = [iid for iid, v in item_latest_map.items() if str(v) in set(map(str, preds))]
                    if not candidates:
                        recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                        st.info("No matching items for predicted properties — fallback to content-based")
                        st.write(recs)
                    else:
                        pop = (
                            events[(events["event"] == "view") & (events["itemid"].astype(str).isin(pd.Series(candidates).astype(str)))]
                            ["itemid"].value_counts().index.astype(str).tolist()[:topN]
                        )
                        if not pop:
                            recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                            st.write(recs)
                        else:
                            st.success("Recommendations")
                            st.write(pop)
                            # allow download
                            csv_buf = io.StringIO()
                            pd.DataFrame({"itemid": pop}).to_csv(csv_buf, index=False)
                            st.download_button("Download recommendations (CSV)", data=csv_buf.getvalue(), file_name="recs.csv")
            else:
                recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                st.success(msg)
                st.write(recs)
                csv_buf = io.StringIO()
                pd.DataFrame({"itemid": recs}).to_csv(csv_buf, index=False)
                st.download_button("Download recommendations (CSV)", data=csv_buf.getvalue(), file_name="recs.csv")

with tab3:
    st.subheader("Anomaly detection: identify abnormal users")

    ae_path = st.text_input("Autoencoder model (h5)", value="cnn_ae.h5")
    scaler_path = st.text_input("Scaler pickle (pkl)", value="scaler.pkl")

    use_dl = False
    ae_model = None
    scaler = None

    if TF_AVAILABLE and os.path.exists(ae_path) and os.path.exists(scaler_path):
        try:
            ae_model = tf.keras.models.load_model(ae_path)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            use_dl = True
            st.success("Loaded DL anomaly artifacts")
        except Exception as e:
            logger.exception("AE load failed")
            st.warning(f"Could not load AE artifacts: {e}")
    else:
        st.info("DL artifacts not available — using IsolationForest fallback")

    user_feats = build_user_features(events)
    if user_feats.empty:
        st.info("No user features — not enough event data")
    else:
        if use_dl and scaler is not None and ae_model is not None:
            X = scaler.transform(user_feats.values)
            X_resh = X.reshape((X.shape[0], X.shape[1], 1))
            recon = ae_model.predict(X_resh, verbose=0)
            mse = np.mean((X_resh - recon) ** 2, axis=(1, 2))
            thr = np.percentile(mse, 98)
            user_feats['recon_error'] = mse
            user_feats['outlier'] = (mse > thr).astype(int)
            st.metric("Detected outliers (DL)", int(user_feats['outlier'].sum()))
            fig = px.histogram(mse, nbins=50, title='Reconstruction error')
            fig.add_vline(x=float(thr), line_color='red')
            st.plotly_chart(fig, use_container_width=True)
        else:
            iso = IsolationForest(contamination=0.02, random_state=42)
            preds = iso.fit_predict(user_feats)
            user_feats['outlier'] = (preds == -1).astype(int)
            user_feats['anomaly_score'] = -iso.score_samples(user_feats.drop(columns=['outlier']))
            st.metric("Detected outliers (IForest)", int(user_feats['outlier'].sum()))
            fig = px.histogram(user_feats['anomaly_score'], nbins=50, title='IsolationForest anomaly scores')
            st.plotly_chart(fig, use_container_width=True)

        # Inspect user table and download
        st.dataframe(user_feats.sort_values('anomaly_score' if 'anomaly_score' in user_feats.columns else 'recon_error', ascending=False).head(200))
        csv_buf = io.StringIO()
        user_feats.to_csv(csv_buf)
        st.download_button("Download user features & scores", data=csv_buf.getvalue(), file_name='user_features.csv')

st.markdown("---")
st.caption("© Recommendation System — improved Streamlit app")
