from __future__ import annotations
import sys, json, os, math
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import kociemba

U, R, F, D, L, B = range(6)
COLORS = "URFDLB"
BASIC = ["U","R","F","D","L","B"]
SUFFIXES = ["", "'", "2"]
ALL_MOVES = [b+s for b in BASIC for s in SUFFIXES]

def solved_facelets() -> str:
    return "".join(c * 9 for c in COLORS)

def cycle(facelets, idxs):
    tmp = facelets[idxs[-1]]
    for i in reversed(range(1, len(idxs))):
        facelets[idxs[i]] = facelets[idxs[i-1]]
    facelets[idxs[0]] = tmp

def apply_perm(s: str, perm: List[Tuple[int,...]], turns: int = 1) -> str:
    f = list(s)
    for _ in range(turns):
        for cyc in perm:
            cycle(f, cyc)
    return "".join(f)

def face_indices(face: int) -> List[int]:
    start = face * 9
    return list(range(start, start + 9))

U_CYCLES = [
    (0, 2, 8, 6), (1, 5, 7, 3),
    (9, 18, 36, 45), (10, 19, 37, 46), (11, 20, 38, 47)
]
R_CYCLES = [
    (9, 11, 17, 15), (10, 14, 16, 12),
    (2, 20, 27, 47), (5, 23, 30, 50), (8, 26, 33, 53)
]
F_CYCLES = [
    (18, 20, 26, 24), (19, 23, 25, 21),
    (6, 36, 27, 11), (7, 39, 28, 10), (8, 42, 29, 9)
]
D_CYCLES = [
    (27, 29, 35, 33), (28, 32, 34, 30),
    (15, 24, 42, 51), (16, 25, 43, 52), (17, 26, 44, 53)
]
L_CYCLES = [
    (36, 38, 44, 42), (37, 41, 43, 39),
    (0, 9, 27, 45), (3, 12, 30, 48), (6, 15, 33, 51)
]
B_CYCLES = [
    (45, 47, 53, 51), (46, 50, 52, 48),
    (0, 11, 35, 38), (1, 10, 34, 37), (2, 9, 33, 36)
]
MOVE_TO_CYCLES = {"U":U_CYCLES,"R":R_CYCLES,"F":F_CYCLES,"D":D_CYCLES,"L":L_CYCLES,"B":B_CYCLES}

def apply_move(facelets: str, move: str) -> str:
    base = move[0]
    suf = move[1:] if len(move) > 1 else ""
    turns = 1 if suf == "" else (2 if suf == "2" else 3)
    return apply_perm(facelets, MOVE_TO_CYCLES[base], turns=turns)

def is_solved(facelets: str) -> bool:
    for f in range(6):
        fi = face_indices(f)
        face = [facelets[i] for i in fi]
        if any(ch != face[0] for ch in face):
            return False
    return True

COLOR_TO_IDX = {c:i for i,c in enumerate(COLORS)}
def encode_facelets(facelets: str):
    if not TORCH_AVAILABLE:
        return None
    x = torch.zeros(54, 6, dtype=torch.float32)
    for i, ch in enumerate(facelets):
        x[i, COLOR_TO_IDX[ch]] = 1.0
    return x.view(-1)

INV = {"U":"U'","U'":"U","U2":"U2",
       "R":"R'","R'":"R","R2":"R2",
       "F":"F'","F'":"F","F2":"F2",
       "D":"D'","D'":"D","D2":"D2",
       "L":"L'","L'":"L","L2":"L2",
       "B":"B'","B'":"B","B2":"B2"}
AXIS = {m:m[0] for m in ALL_MOVES}

def kociemba_solve(facelets: str) -> List[str]:
    sol = kociemba.solve(facelets)
    return sol.split()

if TORCH_AVAILABLE:
    class PolicyNet(nn.Module):
        def __init__(self, hidden=512, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(324, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, len(ALL_MOVES))
            )
        def forward(self, x):
            return self.net(x)
else:
    PolicyNet = None

@torch.no_grad()
def policy_beam_solve(start: str, model_path="policy.pt", beam_size=64, max_depth=30, device="cpu") -> Optional[List[str]]:
    if not TORCH_AVAILABLE or not os.path.exists(model_path):
        return None
    model = PolicyNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    beam = [(start, [], 0.0)]
    seen = {start}

    for _ in range(max_depth+1):
        for s, path, _ in beam:
            if is_solved(s):
                return path
        candidates = []
        for s, path, score in beam:
            x = encode_facelets(s).unsqueeze(0).to(device)
            logits = model(x)[0].cpu()
            probs = torch.softmax(logits, dim=0).numpy()
            order = sorted(range(len(ALL_MOVES)), key=lambda i: -probs[i])
            last = path[-1] if path else None
            last_axis = AXIS[last] if last else None
            last_inv = INV[last] if last else None
            for mi in order:
                mv = ALL_MOVES[mi]
                if last and (mv == last_inv or AXIS[mv] == last_axis):
                    continue
                ns = apply_move(s, mv)
                if ns in seen: continue
                seen.add(ns)
                candidates.append((ns, path + [mv], score + float(math.log(probs[mi] + 1e-9))))
        if not candidates: break
        candidates.sort(key=lambda n: -n[2])
        beam = candidates[:beam_size]
    return None

def solve_facelets(facelets: str, use_ml: bool=False) -> List[str]:
    if use_ml:
        ml = policy_beam_solve(facelets)
        if ml: return ml
    return kociemba_solve(facelets)

def main():
    if "--json" in sys.argv:
        data = sys.stdin.read()
        try:
            req = json.loads(data)
        except Exception:
            print(json.dumps({"error":"invalid json"})); sys.exit(2)
        facelets = req.get("facelets","")
        use_ml = bool(req.get("use_ml", False))
        if not isinstance(facelets, str) or len(facelets) != 54:
            print(json.dumps({"error":"facelets must be a 54-char string"})); sys.exit(3)
        try:
            moves = solve_facelets(facelets, use_ml=use_ml)
            print(json.dumps({"moves": moves, "length": len(moves)})); sys.exit(0)
        except Exception as e:
            print(json.dumps({"error": str(e)})); sys.exit(1)

if __name__ == "__main__":
    main()
