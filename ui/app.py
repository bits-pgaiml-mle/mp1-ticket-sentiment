import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Ticket Sentiment", layout="centered")
st.title("Support Ticket Sentiment")
st.caption("Calls POST /predict on the FastAPI service. No model knowledge in the UI (Teams Lab4 pattern).")

with st.form("ticket_form"):
    text = st.text_area(
        "Ticket / review text",
        value="Support resolved my issue quickly, thank you!",
        height=120,
    )
    channel = st.selectbox("Channel", ["email", "chat", "app"], index=1)
    submitted = st.form_submit_button("Predict sentiment")

if submitted:
    payload = {"text": text, "channel": channel}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach {API_URL} — is `uvicorn serving.api:app --port 8000` running?")
        st.stop()

    if r.status_code == 200:
        result = r.json()
        label = result["label"]
        colour = {"negative": "red", "neutral": "orange", "positive": "green"}.get(label, "blue")
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.markdown(f"Predicted label: **:{colour}[{label}]**")
        st.caption(f"Served by model version: {result['model_version']}")
    else:
        st.error(f"API rejected the request ({r.status_code})")
        try:
            st.json(r.json())
        except Exception:
            st.write(r.text)
