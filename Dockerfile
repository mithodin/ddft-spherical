FROM python:3.9-alpine

COPY Pipfile Pipfile.lock /app/
WORKDIR /app
RUN pip --disable-pip-version-check install pipenv

RUN apk add --no-cache \
        libressl \
        libressl-dev \
        musl-dev \
        libffi-dev \
        py3-pybind11 \
        py3-pybind11-dev \
        openblas \
        openblas-dev \
        libstdc++ \
        alpine-sdk
RUN pipenv install --system --deploy
RUN apk del \
        libressl-dev \
        musl-dev \
        libffi-dev \
        py3-pybind11-dev \
        openblas-dev \
        alpine-sdk

COPY . /app

CMD ["./run.sh", "$DDFT_CONFIG"]
