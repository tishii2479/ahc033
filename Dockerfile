FROM --platform=linux/x86_64 rust:1.70

RUN apt-get update
RUN apt-get install -y vim

WORKDIR /usr/src/solver
COPY src src
COPY tools tools
COPY ahc-utils ahc-utils
COPY Cargo.toml .

# install python
ENV HOME="/root"
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN eval "$(pyenv init -)"
RUN pyenv install 3.10.4
RUN pyenv global 3.10.4

# install python libraries
RUN pip install --no-cache-dir pandas joblib
RUN chmod +x ahc-utils/setup.sh && ./ahc-utils/setup.sh
