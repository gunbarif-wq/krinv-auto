FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default: 5-minute bars, defense 5 symbols, dry-run.
CMD ["python", "realtime_paper_trader.py", "--bar-minutes", "5", "--short", "5", "--long", "20", "--buy-cash", "1000000", "--interval-sec", "30", "--retry", "3", "--throttle-ms", "800", "--dry-run"]
