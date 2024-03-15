FROM tensorflow/tensorflow:2.15.0-gpu

RUN apt-get update \
    && apt-get install -y neovim \
    && apt-get install -y --no-install-recommends gcc git \
    && apt-get install --no-install-recommends --no-install-suggests --assume-yes curl

ENV PATH="/root/.local/bin:$PATH"

# Setup poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.poetry/bin"

# Setup poetry
COPY pyproject.toml /imgalaxy/pyproject.toml
COPY poetry.lock /imgalaxy/poetry.lock
WORKDIR /imgalaxy
COPY . /imgalaxy/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --without dev
# CMD tail -f  /dev/null
ENTRYPOINT ["poetry", "run", "python", "imgalaxy/train.py"]
