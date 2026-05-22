FROM python:3.14-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

CMD ['uv', 'streamlit', 'run', 'app.py', '--server.address = 0.0.0.0', '--server.port = 8501']