"""YouTube Video Performance Predictor - Streamlit front end.

Run with:  streamlit run app.py
The app never trains anything: it collects inputs, sends them to R
(via r_bridge.py) and shows the predictions from the saved models.
"""
import math

import streamlit as st

from r_bridge import load_choices, load_metrics, models_ready, predict

SHORTS_MAX_SECONDS = 60  
WIDE_RANGE_RATIO = 20    
ACCENT = "#4c8bf5"       
REF_COLOR = "#f5a623"    

st.set_page_config(
    page_title="YouTube Performance Predictor", page_icon="📈", layout="wide"
)


def short_number(n):
    """1733046 -> '1.7M', 15000 -> '15K', 100 -> '100'."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= limit:
            return f"{n / limit:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{n:.0f}"


def range_bar_html(low, typical, high, reference=None):
    """A horizontal bar on a log scale: shaded range, a marker for the typical
    value and, optionally, a dashed line for the user's own average."""
    values = [max(v, 1) for v in (low, typical, high)]
    if reference:
        values.append(max(reference, 1))
    first_decade = math.floor(math.log10(min(values)))
    last_decade = math.ceil(math.log10(max(values)))
    if last_decade == first_decade:
        last_decade += 1
    span = last_decade - first_decade

    def pos(x):
        p = (math.log10(max(x, 1)) - first_decade) / span * 100
        return min(max(p, 0), 100)

    p_low, p_typ, p_high = pos(low), pos(typical), pos(high)
    parts = [
        '<div style="margin:12px 30px 36px 30px;">',
        '<div style="position:relative;height:10px;border-radius:6px;'
        'background:rgba(128,128,128,0.28);">',
        f'<div style="position:absolute;left:{p_low:.2f}%;'
        f'width:{max(p_high - p_low, 0.8):.2f}%;height:100%;border-radius:6px;'
        f'background:{ACCENT};opacity:0.55;"></div>',
        f'<div style="position:absolute;left:{p_typ:.2f}%;top:-5px;width:4px;'
        f'height:20px;border-radius:2px;background:{ACCENT};'
        'transform:translateX(-50%);"></div>',
    ]
    if reference:
        parts.append(
            f'<div style="position:absolute;left:{pos(reference):.2f}%;top:-8px;'
            f'height:26px;border-left:2px dashed {REF_COLOR};"></div>'
        )
    for decade in range(first_decade, last_decade + 1):
        left = (decade - first_decade) / span * 100
        parts.append(
            f'<div style="position:absolute;left:{left:.2f}%;top:18px;'
            'transform:translateX(-50%);font-size:11px;opacity:0.65;'
            f'white-space:nowrap;">{short_number(10 ** decade)}</div>'
        )
    parts.append("</div></div>")
    return "".join(parts)


def render_forecast(last):
    """Show the results of the last prediction."""
    result = last["result"]
    payload = last["payload"]
    avg_views = last["avg_views"]

    views, views_low, views_high = (
        result["predicted_views"],
        result["predicted_views_low"],
        result["predicted_views_high"],
    )
    comments, comments_low, comments_high = (
        result["predicted_comments"],
        result["predicted_comments_low"],
        result["predicted_comments_high"],
    )
    video_type = "Short" if payload["duration_sec"] <= SHORTS_MAX_SECONDS else "Long-form"

    views_help = (
        "A central estimate for videos like this in the training data. "
        "Real results vary a lot, so look at the likely range below."
    )
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        delta = None
        if avg_views > 0:
            delta = f"{(views / avg_views - 1) * 100:+.0f}% vs your average"
        st.metric("Typical views", f"{views:,}", delta, help=views_help)
    with c2.container(border=True):
        st.metric("Typical comments", f"{comments:,}")
    with c3.container(border=True):
        if avg_views > 0:
            ratio = views / avg_views
            if ratio >= 1.10:
                verdict = "Above average"
            elif ratio <= 0.90:
                verdict = "Below average"
            else:
                verdict = "About average"
            st.metric("Compared with your average", verdict)
        else:
            st.metric("Video type", video_type)

    with st.container(border=True):
        st.subheader("Likely range")
        st.caption("80% of similar videos in the dataset fall inside the shaded bar.")

        st.markdown(f"**Views:** {views_low:,} to {views_high:,}")
        st.markdown(
            range_bar_html(views_low, views, views_high,
                           avg_views if avg_views > 0 else None),
            unsafe_allow_html=True,
        )
        st.markdown(f"**Comments:** {comments_low:,} to {comments_high:,}")
        st.markdown(
            range_bar_html(comments_low, comments, comments_high),
            unsafe_allow_html=True,
        )

        legend = (
            "Shaded bar = likely range · solid marker = typical value · "
            "the scale is logarithmic (each tick is 10× the previous one)"
        )
        if avg_views > 0:
            legend += " · dashed orange line = your average views"
        st.caption(legend)

        if views_high / max(views_low, 1) >= WIDE_RANGE_RATIO:
            st.warning(
                "The range is very wide, so the model is not confident for this "
                "kind of video. Treat the typical value as a rough guide only."
            )

    details = [
        f"Video type: {video_type}",
        f"Title length: {payload['title_length']} characters",
    ]
    if views > 0:
        details.append(f"Comments per 1,000 views: {comments / views * 1000:.1f}")
    st.caption(" · ".join(details))

    st.info(
        "These numbers describe videos similar to yours in the training "
        "dataset, which mostly contains videos that already got many views. "
        "They are not a guarantee. The model does not know how old a video "
        "is, its thumbnail, or how YouTube's recommendation system will "
        "treat it."
    )

st.title("📈 YouTube Video Performance Predictor")
st.caption(
    "Estimate typical views and comments for a planned video, based on a model "
    "trained in R on a YouTube dataset."
)

if not models_ready():
    st.error(
        "No trained models found. Open a terminal in the project folder, run "
        "`Rscript R/02_train.R`, then refresh this page."
    )
    st.stop()

categories = load_choices()["category"]

with st.sidebar:
    st.header("Video details")
    with st.form("inputs"):
        st.markdown("**Your channel**")
        subscribers = st.number_input("Subscribers", min_value=1, value=10000, step=1000)
        avg_views = st.number_input(
            "Average views of your last 5 videos (optional)",
            min_value=0,
            value=0,
            step=500,
            help="Not used by the model. Only used to compare the prediction "
            "with your usual performance. Leave at 0 to skip.",
        )

        st.markdown("**Planned video**")
        category = st.selectbox("Category", categories)
        title = st.text_input("Title", placeholder="e.g. Easy chocolate cookie recipe")
        col1, col2 = st.columns(2)
        minutes = col1.number_input("Minutes", min_value=0, value=10, step=1)
        seconds = col2.number_input("Seconds", min_value=0, max_value=59, value=0, step=1)

        submitted = st.form_submit_button("Predict performance", type="primary")

# Run the prediction when the form is submitted
error = None
if submitted:
    duration_sec = int(minutes) * 60 + int(seconds)
    if duration_sec <= 0:
        error = "Please enter a duration longer than 0 seconds."
    elif not title.strip():
        error = "Please enter a title. Its length is used by the model."
    else:
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
            st.session_state["last"] = {
                "payload": payload,
                "result": result,
                "avg_views": int(avg_views),
            }
        except RuntimeError as err:
            error = f"Prediction failed: {err}"

# Main area 
tab_forecast, tab_about = st.tabs(["Forecast", "About the model"])

with tab_forecast:
    last = st.session_state.get("last")
    if error:
        st.error(error)
    elif last is None:
        st.info(
            "Fill in the details in the sidebar on the left and press "
            "**Predict performance**."
        )
    else:
        render_forecast(last)

with tab_about:
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