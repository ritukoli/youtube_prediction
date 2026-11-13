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
ABOVE_RATIO = 1.10
BELOW_RATIO = 0.90
ACCENT = "#0e8a75"      # keep in sync with primaryColor in .streamlit/config.toml
REF_COLOR = "#f5a623"

st.set_page_config(
    page_title="YouTube Performance Predictor", page_icon="📈", layout="wide"
)

# No blank lines inside this block: Streamlit's markdown parser would end the HTML there.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, textarea, select,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    font-family: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 2.5rem; max-width: 1100px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
.headline { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; margin-top: 0.5rem; }
.subline { opacity: 0.75; max-width: 60ch; margin: 0.35rem 0 1.25rem 0; }
[data-testid="stMetricValue"] { font-size: 2.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }
</style>
""",
    unsafe_allow_html=True,
)


def short_number(n):
    """1733046 -> '1.7M', 15000 -> '15K', 100 -> '100'."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= limit:
            return f"{n / limit:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{n:.0f}"


def range_bar_html(low, typical, high, reference=None):
    """A horizontal bar on a log scale, labelled directly: the likely range with
    its low and high values, the typical value above the bar and, optionally,
    a dashed line for the user's own average."""
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
    label = "position:absolute;font-size:12px;white-space:nowrap;"
    bottom = 64 if reference else 40
    parts = [
        f'<div role="img" aria-label="Likely range {short_number(low)} to '
        f'{short_number(high)}, typical {short_number(typical)}" '
        f'style="position:relative;height:10px;border-radius:6px;'
        f'margin:44px 44px {bottom}px 44px;background:rgba(128,128,128,0.25);">',
        f'<div style="position:absolute;left:{p_low:.2f}%;'
        f'width:{max(p_high - p_low, 0.8):.2f}%;height:100%;border-radius:6px;'
        f'background:{ACCENT};opacity:0.55;"></div>',
        f'<div style="position:absolute;left:{p_typ:.2f}%;top:-5px;width:4px;'
        f'height:20px;border-radius:2px;background:{ACCENT};'
        'transform:translateX(-50%);"></div>',
        f'<div style="{label}left:{p_typ:.2f}%;top:-30px;'
        f'transform:translateX(-50%);font-weight:600;">{short_number(typical)}</div>',
        f'<div style="{label}top:18px;right:{100 - p_low:.2f}%;'
        f'padding-right:6px;opacity:0.8;">{short_number(low)}</div>',
        f'<div style="{label}top:18px;left:{p_high:.2f}%;'
        f'padding-left:6px;opacity:0.8;">{short_number(high)}</div>',
    ]
    if reference:
        p_ref = pos(reference)
        parts.append(
            f'<div style="position:absolute;left:{p_ref:.2f}%;top:-8px;'
            f'height:46px;border-left:2px dashed {REF_COLOR};"></div>'
        )
        parts.append(
            f'<div style="{label}left:{p_ref:.2f}%;top:42px;'
            f'transform:translateX(-50%);opacity:0.8;">'
            f'Your average {short_number(reference)}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_forecast(last):
    """Show the results of the last prediction."""
    result = last["result"]
    payload = last["payload"]
    avg_views = last["avg_views"]
    has_avg = avg_views > 0

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
    avg_likes = last.get("avg_likes", 0)
    has_likes = "predicted_likes" in result
    if has_likes:
        likes, likes_low, likes_high = (
            result["predicted_likes"],
            result["predicted_likes_low"],
            result["predicted_likes_high"],
        )
    video_type = "Short" if payload["duration_sec"] <= SHORTS_MAX_SECONDS else "Long-form"
    ratio = views / avg_views if has_avg else None

    if ratio is None:
        headline = "Here's how similar videos performed"
    elif ratio >= ABOVE_RATIO:
        headline = "Likely to beat your average"
    elif ratio <= BELOW_RATIO:
        headline = "Likely to land below your average"
    else:
        headline = "Likely close to your average"

    subline = (
        f"80% of similar videos get between {short_number(views_low)} and "
        f"{short_number(views_high)} views."
    )
    if has_avg:
        subline += f" Your average is {short_number(avg_views)}."

    st.markdown(f'<div class="headline">{headline}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subline">{subline}</div>', unsafe_allow_html=True)

    col_views, col_likes, col_comments = st.columns(3)
    with col_views.container(border=True):
        delta = f"{(ratio - 1) * 100:+.0f}% vs your average" if has_avg else None
        st.metric(
            "Typical views",
            short_number(views),
            delta,
            help=f"Exact estimate: {views:,}. A central estimate for videos like "
            "this in the training data. Real results vary a lot, so look at the "
            "range below it.",
        )
        st.markdown(
            range_bar_html(views_low, views, views_high, avg_views if has_avg else None),
            unsafe_allow_html=True,
        )
    with col_likes.container(border=True):
        if has_likes:
            likes_delta = (
                f"{(likes / avg_likes - 1) * 100:+.0f}% vs your average"
                if avg_likes > 0 else None
            )
            st.metric(
                "Typical likes",
                short_number(likes),
                likes_delta,
                help=f"Exact estimate: {likes:,}. Look at the range below it for "
                "what to expect.",
            )
            st.markdown(
                range_bar_html(likes_low, likes, likes_high,
                               avg_likes if avg_likes > 0 else None),
                unsafe_allow_html=True,
            )
        else:
            st.info("No likes model yet. Run `Rscript R/02_train.R` again to add it.")
    with col_comments.container(border=True):
        st.metric(
            "Typical comments",
            short_number(comments),
            help=f"Exact estimate: {comments:,}. Look at the range below it for "
            "what to expect.",
        )
        st.markdown(
            range_bar_html(comments_low, comments, comments_high),
            unsafe_allow_html=True,
        )

    legend = "Shaded bar: the range 80% of similar videos fall in. Solid marker: typical value."
    if has_avg or (has_likes and avg_likes > 0):
        legend += " Dashed line: your own average."
    legend += " The scale is logarithmic, so each step is about 10× the last."
    st.caption(legend)

    if views_high / max(views_low, 1) >= WIDE_RANGE_RATIO:
        st.warning(
            "The range is very wide, so the model is not confident for this "
            "kind of video. Treat the typical value as a rough guide only."
        )

    details = [
        video_type,
        f"{payload['title_length']}-character title",
    ]
    if views > 0:
        if has_likes:
            details.append(f"{likes / views * 100:.1f} likes per 100 views")
        details.append(f"{comments / views * 1000:.1f} comments per 1,000 views")
    st.caption(" · ".join(details))

    with st.expander("What this estimate doesn't cover"):
        st.write(
            "These numbers describe videos similar to yours in the training "
            "dataset, which mostly contains videos that already got many views. "
            "They are not a guarantee. The model does not know how old a video "
            "is, its thumbnail, or how YouTube's recommendation system will "
            "treat it."
        )


st.title("YouTube performance predictor")
st.caption(
    "Estimate typical views and comments for a planned video, using a model "
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

        avg_likes = st.number_input(
            "Average likes of your last 5 videos (optional)",
            min_value=0,
            value=0,
            step=50,
            help="Not used by the model. Only used to compare the predicted "
            "likes with your usual performance. Leave at 0 to skip.",
        )

        st.markdown("**Planned video**")
        category = st.selectbox("Category", categories)
        title = st.text_input(
            "Title",
            placeholder="e.g. Easy chocolate cookie recipe",
            help="The model uses the length of the title, not its words.",
        )
        col1, col2 = st.columns(2)
        minutes = col1.number_input("Minutes", min_value=0, value=10, step=1)
        seconds = col2.number_input("Seconds", min_value=0, max_value=59, value=0, step=1)

        submitted = st.form_submit_button("Predict performance", type="primary")

# Run the prediction when the form is submitted
error = None
if submitted:
    duration_sec = int(minutes) * 60 + int(seconds)
    if duration_sec <= 0:
        error = "Enter a duration longer than 0 seconds."
    elif not title.strip():
        error = "Enter a title. Its length is used by the model."
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
                "avg_likes": int(avg_likes),
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
        with st.container(border=True):
            st.markdown("#### Plan a video, then see its likely reach")
            st.write(
                "Enter your subscriber count and the video's category, title and "
                "length in the sidebar. You'll get typical views and comments, "
                "plus the range most similar videos fall in."
            )
            st.caption("On a phone, open the sidebar with the arrow at the top left.")
    else:
        render_forecast(last)

with tab_about:
    st.write(
        "Two models are trained in R for each target (views, likes and comments): a "
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
        st.write("Evaluation on test data (MAE and RMSE are in views, likes or comments):")
        st.dataframe(metrics, hide_index=True)