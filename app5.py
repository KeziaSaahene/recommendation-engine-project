import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
-------------------------
Page Config + Styling
-------------------------
st.set_page_config(page_title="📊 Text Classification + Recommendations", layout="wide")
st.markdown("""
<style>
@keyframes fadeIn {
from {opacity: 0; transform: translateY(-10px);}
to {opacity: 1; transform: translateY(0);}
}
.intro-box {
background-color: #f0f4ff;
padding: 15px 20px;
border-radius: 12px;
margin-bottom: 20px;
animation: fadeIn 1.5s ease-in-out;
color: #3B3B98;
font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
color: #3B3B98;
font-family: 'Poppins', sans-serif;
}
.stButton>button {
background-color: #3B3B98;
color: white;
border-radius: 10px;
padding: 0.5rem 1rem;
font-weight: 500;
border: none;
transition: 0.3s;
}
.stButton>button:hover {
background-color: #575fcf;
transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)
st.title("📊 Text Classification & Recommendation System")
st.markdown("""
<div class="intro-box"> ✨ Welcome! This app combines text classification and recommendation systems. Upload your data, choose a user, and discover intelligent insights powered by deep learning. </div> """, unsafe_allow_html=True)
-------------------------
Load CSV data
-------------------------
st.sidebar.header("📦 Data Upload")
events_file = st.sidebar.file_uploader("events.csv", type=["csv"])
props_file = st.sidebar.file_uploader("item_properties.csv", type=["csv"])
def load_csv(file, default_name):
if file is not None:
return pd.read_csv(file)
elif os.path.exists(default_name):
return pd.read_csv(default_name)
else:
st.error(f"Missing {default_name}")
return pd.DataFrame()
events = load_csv(events_file, "events.csv")
item_props = load_csv(props_file, "item_properties.csv")
if events.empty or item_props.empty:
st.stop()
-------------------------
Clean data
-------------------------
def ensure_datetime_ms(series):
def _to_dt(x):
try:
xv = float(x)
if xv > 1e12:
return pd.to_datetime(int(xv), unit="ms", errors="coerce")
elif xv > 1e10:
return pd.to_datetime(int(xv), unit="us", errors="coerce")
else:
return pd.to_datetime(int(xv), unit="s", errors="coerce")
except:
return pd.to_datetime(x, errors="coerce")
return series.apply(_to_dt)
events["timestamp"] = ensure_datetime_ms(events["timestamp"])
item_props["timestamp"] = ensure_datetime_ms(item_props["timestamp"])
events = events.dropna(subset=["timestamp", "visitorid", "event", "itemid"])
item_props = item_props.dropna(subset=["timestamp", "itemid", "property", "value"])
Convert numeric strings
def clean_numeric_str(val):
try:
return float(str(val).replace("n","").split()[-1])
except:
return val
item_props["value"] = item_props["value"].apply(clean_numeric_str)
-------------------------
Load Task 1 CNN artifacts
-------------------------
cnn_model = None
tokenizer = None
label_classes = None
MAX_LEN = 100
try:
cnn_model = tf.keras.models.load_model("cnn_model.keras")
with open("tokenizer.json", "r") as f:
tok_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tok_data))
label_classes = np.load("label_classes.npy", allow_pickle=True)
st.success("CNN artifacts loaded successfully!")
except Exception as e:
st.warning(f"Could not load CNN artifacts: {e}")
-------------------------
Helper functions
-------------------------
def latest_item_property(item_props, prop_name):
sub = item_props[item_props["property"]==prop_name].sort_values("timestamp")
return sub.groupby("itemid")["value"].last().to_dict()
def recommend_content_based(events, item_props, visitorid, topn=10):
noisy = {"available", "categoryid"}
props_pivot = (
item_props[~item_props["property"].isin(noisy)]
.sort_values("timestamp")
.groupby(["itemid", "property"])["value"].last().unstack(fill_value="")
)
props_pivot = props_pivot.fillna("")
props_pivot["text"] = props_pivot.astype(str).apply(lambda r: " ".join(r.values.tolist()), axis=1)
vect = TfidfVectorizer(max_features=5000)
item_tfidf = vect.fit_transform(props_pivot["text"].values)
uviews = events[(events["visitorid"]==visitorid) & (events["event"]=="view")]
viewed_ids = [str(i) for i in uviews["itemid"].tolist()]
if len(viewed_ids)==0:
pop = events[events["event"]=="view"]["itemid"].value_counts().index[:topn].astype(str).tolist()
return pop, "Cold-start fallback: popular items."
sub = props_pivot.loc[props_pivot.index.astype(str).isin(viewed_ids)]
if sub.empty:
pop = events[events["event"]=="view"]["itemid"].value_counts().index[:topn].astype(str).tolist()
return pop, "No overlap with properties; popular items."
sub_tfidf = vect.transform(sub["text"].values)
user_vec = sub_tfidf.mean(axis=0)
sims = cosine_similarity(user_vec, item_tfidf).ravel()
props_pivot = props_pivot.assign(similarity=sims)
props_pivot = props_pivot.loc[~props_pivot.index.astype(str).isin(viewed_ids)]
recs = props_pivot.sort_values("similarity", ascending=False).index.astype(str).tolist()[:topn]
return recs, "Content-based recommendations."
-------------------------
Task 1 UI
-------------------------
st.subheader("📝 Text Classification & Recommendations (Task 1)")
visitor_ids = events["visitorid"].unique().tolist()
chosen_user = st.selectbox("Select a visitor", visitor_ids[:5000] if len(visitor_ids)>5000 else visitor_ids)
topN = st.slider("Number of recommendations", 5, 30, 10)
if st.button("Classify & Recommend"):
if cnn_model and tokenizer is not None and label_classes is not None:
target_prop = item_props["property"].value_counts().index[0]
item_latest = latest_item_property(item_props, target_prop)
hist = events[(events["visitorid"]==chosen_user) & (events["event"]=="view")].sort_values("timestamp")
tokens = [str(item_latest.get(i,"")) for i in hist["itemid"].values if str(item_latest.get(i,""))!=""]
if len(tokens)==0:
st.info("No valid tokens from user's history; fallback content-based.")
recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
st.success(msg)
st.write("Recommendations (itemid):", recs)
else:
text = " ".join(tokens)
seq = tokenizer.texts_to_sequences([text])
pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
probs = cnn_model.predict(pad, verbose=0)[0]
top_idx = np.argsort(probs)[::-1][:3]
predicted_props = [label_classes[i] for i in top_idx]
st.write("Top predicted property values:", predicted_props)
target_items = [iid for iid, val in item_latest.items() if str(val) in set(map(str, predicted_props))]
if len(target_items)==0:
recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
st.success("No direct matches. " + msg)
st.write("Recommendations (itemid):", recs)
else:
pop = (events[(events["event"]=="view") & (events["itemid"].astype(str).isin(pd.Series(target_items).astype(str)))]
["itemid"].value_counts().index.astype(str).tolist()[:topN])
if len(pop)==0:
recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
st.success("Sparse segment. " + msg)
st.write("Recommendations (itemid):", recs)
else:
st.success("CNN-driven property match recommendations:")
st.write("Recommendations (itemid):", pop)
else:
recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
st.success(msg)
st.write("Recommendations (itemid):", recs)
-------------------------
Task 2 UI: Anomaly Detection
-------------------------
st.subheader("🚨 User Anomaly Detection (Task 2)")
def build_user_features(events_df):
feats = []
for uid, g in events_df.groupby("visitorid"):
total = len(g)
views = (g["event"]=="view").sum()
adds = (g["event"]=="addtocart").sum()
buys = (g["event"]=="transaction").sum()
feats.append({
"visitorid": uid,
"total_events": total,
"views": views,
"adds": adds,
"buys": buys,
"add_rate": (adds/total) if total>0 else 0.0,
"conv_rate": (buys/total) if total>0 else 0.0
})
return pd.DataFrame(feats).set_index("visitorid")
user_feats = build_user_features(events)
ae_model = None
scaler = None
try:
ae_model = tf.keras.models.load_model("cnn_ae.keras")
with open("scaler.pkl","rb") as f:
scaler = pickle.load(f)
except Exception:
pass
if st.button("Detect Outliers"):
if ae_model and scaler:
X_scaled = scaler.transform(user_feats.values)
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
recon = ae_model.predict(X_cnn, verbose=0)
mse = np.mean((X_cnn - recon)**2, axis=(1,2))
threshold = np.percentile(mse, 98)
user_feats["recon_error"] = mse
user_feats["outlier"] = (mse>threshold).astype(int)
st.metric("Detected outliers", int(user_feats["outlier"].sum()))
st.line_chart(mse)
else:
iso = IsolationForest(contamination=0.02, random_state=42)
preds = iso.fit_predict(user_feats)
user_feats["outlier"] = (preds==-1).astype(int)
st.metric("Detected outliers (Isolation Forest)", int(user_feats["outlier"].sum()))
scores = -iso.score_samples(user_feats.drop(columns=["outlier"]))
st.line_chart(scores)
