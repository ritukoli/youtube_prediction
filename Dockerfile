# R and Python in one image, so r_bridge.py can find Rscript.
# Use the same R version as on your machine (R --version) so the saved models load.
FROM rocker/r-ver:4.6.2

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*
RUN Rscript -e 'install.packages(c("ranger", "jsonlite"))'

WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

COPY . .
# Optional but safest: retrain inside the image so the models match its R packages
# (needs data/youtube_dataset.csv in the repo)
RUN Rscript R/02_train.R

# Change app/app.py if your entry file is elsewhere. Hugging Face Spaces uses port 7860.
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860}
