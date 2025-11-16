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

# Python + venv tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a venv and install packages into it
ENV PY_HOME=/opt/py
RUN python3 -m venv $PY_HOME

# Torch optional (large). Install into venv.
ARG ENABLE_ML=false
ENV ENABLE_ML=${ENABLE_ML}

RUN $PY_HOME/bin/pip install --no-cache-dir kociemba tqdm && \
    if [ "$ENABLE_ML" = "true" ]; then \
        $PY_HOME/bin/pip install --no-cache-dir \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Make the venv Python first on PATH so "python3" resolves to it
ENV PATH="$PY_HOME/bin:${PATH}"

WORKDIR /app
COPY --from=build /app/publish ./
COPY ./python ./python

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "WebRubikSolver.dll"]
