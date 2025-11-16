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

# Install Python + pip
RUN apt-get update &&     apt-get install -y --no-install-recommends python3 python3-pip &&     rm -rf /var/lib/apt/lists/*

# Install required Python packages
# Torch is optional because of image size; controlled via build-arg
ARG ENABLE_ML=false
ENV ENABLE_ML=${ENABLE_ML}

RUN pip3 install --no-cache-dir kociemba tqdm &&     if [ "$ENABLE_ML" = "true" ]; then         pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ;     fi

WORKDIR /app
COPY --from=build /app/publish ./
COPY ./python ./python

# App settings
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "WebRubikSolver.dll"]
