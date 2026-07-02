FROM python:3.12.3-slim AS builder

# 2. uvの公式イメージからバイナリだけをコピー（非常に高速・軽量）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 3. 依存関係ファイルのコピーとキャッシュを利用したインストール
# (ロックファイルが変わらない限り、このレイヤーはキャッシュされます)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 4. アプリケーションコードのコピーとプロジェクト自体のインストール
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 5. 実行用の軽量なステージ
FROM python:3.12.3-slim

WORKDIR /app

# builderから仮想環境（.venv）だけをコピー
COPY --from=builder /app/.venv /app/.venv

# 仮想環境のパスを環境変数に通す（これで`python`や`uvicorn`がそのまま動きます）
ENV PATH="/app/.venv/bin:$PATH"

# アプリケーションコードのコピー
COPY . /app

CMD ["python", "main.py"]