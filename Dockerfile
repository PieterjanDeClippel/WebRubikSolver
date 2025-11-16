# =========================
# docker build -t rubik-solver:lite .
# docker run --rm -p 8080:8080 rubik-solver:lite
# Open http://localhost:8080

# Example
# UUFUBDFLL
# RDFRURRBB
# DFLLFURDF
# UDDUBLRFU
# LRLFLDDRL
# BBLBBLUBF

# =========================


# =========================
# Build stage (dotnet)
# =========================
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY ./src/WebRubikSolver/WebRubikSolver.csproj ./src/WebRubikSolver/WebRubikSolver.csproj
RUN dotnet restore ./src/WebRubikSolver/WebRubikSolver.csproj

COPY ./src/WebRubikSolver ./src/WebRubikSolver
RUN dotnet publish ./src/WebRubikSolver/WebRubikSolver.csproj -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:8.0

# Python + build deps (for kociemba C extension)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a venv and upgrade packaging tools
ENV PY_HOME=/opt/py
RUN python3 -m venv $PY_HOME && \
    $PY_HOME/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Optional ML (large)
ARG ENABLE_ML=false
ENV ENABLE_ML=${ENABLE_ML}

# Install Python deps into venv
RUN $PY_HOME/bin/pip install --no-cache-dir kociemba tqdm && \
    if [ "$ENABLE_ML" = "true" ]; then \
        $PY_HOME/bin/pip install --no-cache-dir \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Make venv python first on PATH
ENV PATH="$PY_HOME/bin:${PATH}"

WORKDIR /app
COPY --from=build /app/publish ./
COPY ./python ./python

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "WebRubikSolver.dll"]
