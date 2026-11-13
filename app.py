"""YouTube Video Performance Predictor - Streamlit front end.

Run with:  streamlit run app.py
The app never trains anything: it collects inputs, sends them to R
(via r_bridge.py) and shows the predictions from the saved models.
"""
import pandas as pd
import streamlit as st

from r_bridge import load_choices, load_metrics, models_ready, predict

SHORTS_MAX_SECONDS = 60  # keep in sync with SHORTS_MAX_SECONDS in R/00_config.R
WIDE_RANGE_RATIO = 20    # warn when the high end is this many times the low end

st.set_page_config(page_title="YouTube Performance Predictor", page_icon="📈")
st.title("YouTube Video Performance Predictor")
st.caption(
    "Estimate typical views and comments for a planned video, based on a model "
    "trained in R on a YouTube dataset."
)

# ---- Make sure the models exist ---------------------------------------------
if not models_ready():
    st.error(
        "No trained models found. Open a terminal in the project folder, run "
        "`Rscript R/02_train.R`, then refresh this page."
    )
    st.stop()

categories = load_choices()["category"]

# ---- Input form -------------------------------------------------------------
with st.form("inputs"):
    st.subheader("Your channel")
    subscribers = st.number_input("Subscribers", min_value=1, value=10000, step=1000)
    avg_views = st.number_input(
        "Average views of your last 5 videos (optional)",
        min_value=0,
        value=0,
        step=500,
        help="Not used by the model. Only used to compare the prediction with "
        "your usual performance. Leave at 0 to skip.",
    )

    st.subheader("Planned video")
    category = st.selectbox("Category", categories)
    title = st.text_input("Title")
    col1, col2 = st.columns(2)
    minutes = col1.number_input("Duration: minutes", min_value=0, value=10, step=1)
    seconds = col2.number_input(
        "Duration: seconds", min_value=0, max_value=59, value=0, step=1
    )

    submitted = st.form_submit_button("Predict performance")

# ---- Prediction -------------------------------------------------------------
if submitted:
    duration_sec = int(minutes) * 60 + int(seconds)
    if duration_sec <= 0:
        st.error("Please enter a duration longer than 0 seconds.")
        st.stop()
    if not title.strip():
        st.error("Please enter a title. Its length is used by the model.")
        st.stop()

    # Feature names must match what R/02_train.R was trained on
    payload = {
        "subscribers": int(subscribers),
        "duration_sec": duration_sec,
        "title_length": len(title.strip()),
        "category": category,
    }

    try:
        with st.spinner("Running the R model..."):
            result = predict(payload)
    except RuntimeError as err:
        st.error(f"Prediction failed: {err}")
        st.stop()

    views = result["predicted_views"]
    views_low = result["predicted_views_low"]
    views_high = result["predicted_views_high"]
    comments = result["predicted_comments"]
    comments_low = result["predicted_comments_low"]
    comments_high = result["predicted_comments_high"]
    video_type = "Short" if duration_sec <= SHORTS_MAX_SECONDS else "Long-form"

    st.divider()
    st.subheader("Forecast")

    views_help = (
        "A central estimate for videos like this in the training data. "
        "Real results vary a lot, so look at the likely range below."
    )
    if avg_views > 0:
        ratio = views / avg_views
        delta = f"{(ratio - 1) * 100:+.0f}% vs your average"
        if ratio >= 1.10:
            verdict = "Above average"
        elif ratio <= 0.90:
            verdict = "Below average"
        else:
            verdict = "About average"
        c1, c2, c3 = st.columns(3)
        c1.metric("Typical views", f"{views:,}", delta, help=views_help)
        c2.metric("Typical comments", f"{comments:,}")
        c3.metric("Compared with your average", verdict)
    else:
        c1, c2 = st.columns(2)
        c1.metric("Typical views", f"{views:,}", help=views_help)
        c2.metric("Typical comments", f"{comments:,}")

    st.markdown(
        "**Likely range** (80% of similar videos in the data fall inside it): "
        f"{views_low:,} to {views_high:,} views, and "
        f"{comments_low:,} to {comments_high:,} comments."
    )
    if views_high / max(views_low, 1) >= WIDE_RANGE_RATIO:
        st.warning(
            "The range is very wide, so the model is not confident for this "
            "kind of video. Treat the typical value as a rough guide only."
        )

    details = [f"Video type: {video_type}", f"Title length: {payload['title_length']} characters"]
    if views > 0:
        details.append(f"Comments per 1,000 views: {comments / views * 1000:.1f}")
    st.caption(" · ".join(details))

    # Numbered labels keep the bars in this order (the chart sorts by name)
    labels = ["1. Low end of range", "2. Typical", "3. High end of range"]
    values = [views_low, views, views_high]
    if avg_views > 0:
        labels.append("4. Your average (last 5)")
        values.append(avg_views)
    st.bar_chart(pd.DataFrame({"Views": values}, index=labels))
    st.caption(
        "Predicted views. The bars use a normal (not log) scale, so when the "
        "range is wide the low end looks tiny next to the high end."
    )

    st.info(
        "These numbers describe videos similar to yours in the training "
        "dataset, which mostly contains videos that already got many views. "
        "They are not a guarantee. The model does not know how old a video "
        "is, its thumbnail, or how YouTube's recommendation system will "
        "treat it."
    )

# ---- About the model --------------------------------------------------------
with st.expander("About the model"):
    st.write(
        "Two models are trained in R for each target (views and comments): a "
        "linear regression baseline and a random forest. The one with the lower "
        "error on held-out channels is used. Inputs: subscribers, video "
        "duration, title length and category."
    )
    st.write(
        "The likely range is the 10th to 90th percentile of outcomes for "
        "similar videos. `Range_coverage` in the table below is the share of "
        "test videos whose real value fell inside that range. A value near 0.8 "
        "means the range is well calibrated."
    )
    metrics = load_metrics()
    if metrics is not None:
        st.write("Evaluation on test data (MAE and RMSE are in views/comments):")
        st.dataframe(metrics, hide_index=True)