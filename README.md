# WebRubikSolver

A Dockerized .NET 8 + Python web app to input Rubik's cube facelets (URFDLB order), solve via Python (Kociemba; optional ML policy-guided beam), and visualize a 3D cube with true layer turns in the browser.

## Build & Run

```bash
docker build -t rubik-solver:lite .
docker run --rm -p 8080:8080 rubik-solver:lite
# open http://localhost:8080
```

Build with optional ML packages:
```bash
docker build --build-arg ENABLE_ML=true -t rubik-solver:ml .
docker run --rm -p 8080:8080 rubik-solver:ml
```
