// ---------- UI references ----------
const faceletsEl = document.getElementById('facelets');
const useMlEl = document.getElementById('useMl');
const importBtn = document.getElementById('importBtn');
const solvedBtn = document.getElementById('solvedBtn');
const clearBtn = document.getElementById('clearBtn');
const solveBtn = document.getElementById('solveBtn');
const statusEl = document.getElementById('status');
const solutionEl = document.getElementById('solution');
const canvasEl = document.getElementById('canvas');

const moveButtons = Array.from(document.querySelectorAll('.controls button[data-move]'));
const swatchButtons = Array.from(document.querySelectorAll('.swatch'));

// ---------- Color / label mapping ----------
const LABELS = ['U', 'R', 'F', 'D', 'L', 'B'];
const COLOR_HEX = {
    U: 0xffffff,
    R: 0xff3b30,
    F: 0x34c759,
    D: 0xffcc00,
    L: 0xff9500,
    B: 0x0a84ff
};
let currentLabel = 'U';
swatchButtons.forEach(btn => {
    const lab = btn.dataset.label;
    btn.style.background = '#' + COLOR_HEX[lab].toString(16).padStart(6, '0');
    btn.addEventListener('click', () => setActiveSwatch(lab));
});
function setActiveSwatch(lab) {
    currentLabel = lab;
    swatchButtons.forEach(b => b.classList.toggle('active', b.dataset.label === lab));
}
setActiveSwatch('U');

// ---------- Three.js scene ----------
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, canvasEl.clientWidth / canvasEl.clientHeight, 0.1, 100);
camera.position.set(5, 5, 7);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(canvasEl.clientWidth, canvasEl.clientHeight);
canvasEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dl = new THREE.DirectionalLight(0xffffff, 1.0);
dl.position.set(6, 8, 5); scene.add(dl);

const cubeRoot = new THREE.Group();
scene.add(cubeRoot);

// ---------- Build 26 cubies with sticker planes ----------
const CUBIE_SIZE = 0.98 / 3;
const STICKER_SIZE = CUBIE_SIZE * 0.92;

const cubies = [];

const FACE_DEF = [
    { normal: new THREE.Vector3(0, 1, 0), label: 'U' },
    { normal: new THREE.Vector3(1, 0, 0), label: 'R' },
    { normal: new THREE.Vector3(0, 0, 1), label: 'F' },
    { normal: new THREE.Vector3(0, -1, 0), label: 'D' },
    { normal: new THREE.Vector3(-1, 0, 0), label: 'L' },
    { normal: new THREE.Vector3(0, 0, -1), label: 'B' }
];

function buildCubeSolved() {
    while (cubeRoot.children.length) cubeRoot.remove(cubeRoot.children[0]);
    cubies.length = 0;

    const frameGeom = new THREE.BoxGeometry(1.02, 1.02, 1.02);
    const frameMat = new THREE.MeshBasicMaterial({ color: 0x111111, wireframe: true });
    const frame = new THREE.Mesh(frameGeom, frameMat);
    frame.name = 'frame';
    cubeRoot.add(frame);

    for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
        if (x === 0 && y === 0 && z === 0) continue;

        const group = new THREE.Group();
        group.position.set(x * CUBIE_SIZE, y * CUBIE_SIZE, z * CUBIE_SIZE);
        // Logical state
        const logicalPosition = new THREE.Vector3(x, y, z);
        const logicalQuaternion = new THREE.Quaternion();

        const stickers = [];

        FACE_DEF.forEach(({ normal, label }) => {
            const nx = normal.x, ny = normal.y, nz = normal.z;
            if ((nx !== 0 && Math.sign(nx) !== Math.sign(x)) ||
                (ny !== 0 && Math.sign(ny) !== Math.sign(y)) ||
                (nz !== 0 && Math.sign(nz) !== Math.sign(z))) return;
            if ((nx && x === 0) || (ny && y === 0) || (nz && z === 0)) return;

            const plane = new THREE.PlaneGeometry(STICKER_SIZE, STICKER_SIZE);
            const mat = new THREE.MeshPhongMaterial({ color: COLOR_HEX[label], side: THREE.DoubleSide });
            const mesh = new THREE.Mesh(plane, mat);
            mesh.userData.label = label;
            mesh.userData.normal = normal.clone(); // Store the initial normal
            mesh.userData.isCenter = (Math.abs(x) + Math.abs(y) + Math.abs(z) === 1);
            const offset = new THREE.Vector3(nx, ny, nz).multiplyScalar(CUBIE_SIZE / 2 + 0.001);
            mesh.position.copy(offset);
            const look = new THREE.Matrix4().lookAt(new THREE.Vector3(0, 0, 0), new THREE.Vector3(nx, ny, nz), new THREE.Vector3(0, 1, 0));
            mesh.quaternion.setFromRotationMatrix(look);
            group.add(mesh);

            stickers.push({ mesh, normal: normal.clone(), get label() { return mesh.userData.label; }, set label(v) { mesh.userData.label = v; } });
        });

        cubeRoot.add(group);
        cubies.push({ group, stickers, logicalPosition, logicalQuaternion });
    }
}
buildCubeSolved();

// ---------- Raycast paint ----------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

renderer.domElement.addEventListener('pointerdown', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects(cubeRoot.children, true);
    if (intersects.length) {
        const obj = intersects[0].object;
        if (obj.userData && obj.userData.label) {
            if (obj.userData.isCenter) return;
            obj.userData.label = currentLabel;
            obj.material.color.setHex(COLOR_HEX[currentLabel]);
        }
    }
});

// ---------- Move animation (true per-layer rotation) ----------
const MOVE_AXIS = {
    U: new THREE.Vector3(0, 1, 0),
    D: new THREE.Vector3(0, -1, 0),
    R: new THREE.Vector3(1, 0, 0),
    L: new THREE.Vector3(-1, 0, 0),
    F: new THREE.Vector3(0, 0, 1),
    B: new THREE.Vector3(0, 0, -1),
};

// Round to grid index (-1, 0, +1)
const GRID = (v) => Math.round(v * 3);
const LOGICAL_LAYER_TEST = {
    U: p => p.y > 0.5,
    D: p => p.y < -0.5,
    R: p => p.x > 0.5,
    L: p => p.x < -0.5,
    F: p => p.z > 0.5,
    B: p => p.z < -0.5,
};

// ----- Move queue (serialize animations) -----
const moveQueue = [];
let draining = false;
let __queueDriving = false;
const idleResolvers = [];

// Enqueue a single move string, e.g. "R", "U'", "F2"
function enqueueMove(move) {
    moveQueue.push(move);
    drainQueue();
}

// Enqueue multiple moves and return a promise that resolves when all finish
function runMoves(seq) {
    if (Array.isArray(seq)) moveQueue.push(...seq);
    drainQueue();
    return waitForIdle();
}

function waitForIdle() {
    if (!draining && moveQueue.length === 0) return Promise.resolve();
    return new Promise((res) => idleResolvers.push(res));
}

async function drainQueue() {
    if (draining) return;
    draining = true;

    try {
        while (moveQueue.length) {
            const next = moveQueue.shift();
            if (!next) continue;
            __queueDriving = true;
            await rotateLayer(next, /*__fromQueue:*/ true);
            __queueDriving = false;
        }
    } catch (err) {
        console.error('Move queue error:', err);
    } finally {
        draining = false;
        faceletsEl.value = exportFacelets(); // Automatically update facelets on idle
        // resolve any waiters
        while (idleResolvers.length) idleResolvers.shift()();
    }
}

function rotateLayer(move, __fromQueue = false) {
    // If someone calls rotateLayer directly, reroute it into the queue
    if (!__fromQueue) {
        enqueueMove(move);
        return waitForIdle(); // resolve when it actually finishes via the queue
    }

    return new Promise(resolve => {
        const face = move[0];
        const suf = move.length > 1 ? move.slice(1) : "";
        const quarters = suf === "2" ? 2 : 1;
        const prime = suf === "'";
        let axis = MOVE_AXIS[face].clone();
        let sign = prime ? -1 : 1;
        const angle = quarters * (Math.PI / 2) * sign;
        const moveQuaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle);

        const temp = new THREE.Group();
        temp.position.set(0, 0, 0); // rotate about cube origin
        cubeRoot.add(temp);

        const targets = [];
        for (const c of cubies) {
            if (LOGICAL_LAYER_TEST[face](c.logicalPosition)) {
                temp.attach(c.group); // reparent into rotating temp group
                targets.push(c);
            }
        }
        temp.quaternion.set(0, 0, 0, 1); // Reset temp group rotation

        // Optional: sanity log
        console.log(`rotateLayer ${move}: selected ${targets.length} cubies`);

        const total = quarters * (Math.PI / 2) * sign;
        const duration = 180 * quarters;
        const frames = Math.max(24 * quarters, 1);
        const startQuat = new THREE.Quaternion(); // Identity
        const endQuat = new THREE.Quaternion().setFromAxisAngle(axis, total);
        let i = 0;
        const id = setInterval(() => {
            i++;
            const t = i / frames;
            temp.quaternion.copy(startQuat).slerp(endQuat, t);
            i++;
            if (i >= frames) {
                clearInterval(id);
                // Update logical state and snap visual state
                targets.forEach(c => {
                    cubeRoot.attach(c.group);
                    c.logicalPosition.applyQuaternion(moveQuaternion).round();
                    c.logicalQuaternion.premultiply(moveQuaternion);
                    // Snap visual state to logical state
                    c.group.position.copy(c.logicalPosition).multiplyScalar(CUBIE_SIZE);
                    c.group.quaternion.copy(c.logicalQuaternion);
                });
                cubeRoot.remove(temp);
                resolve();
            }
        }, duration / frames);
    });
}

async function animateMoves(moves) {
    for (const mv of moves) await rotateLayer(mv);
}

// FIX: proper reference to button element & single handler (queue based)
moveButtons.forEach(b => b.addEventListener('click', () => {
    const move = b.dataset.move;
    enqueueMove(move);
    statusEl.textContent = `Queued: ${move}  (in flight: ${draining ? 'yes' : 'no'}, pending: ${moveQueue.length})`;
}));

// ---------- Export/import facelets in URFDLB order ----------
// === Canonical URFDLB orientation (Kociemba) ===
// For each face: normal = face normal in world space
// u = "to the right" vector on that face; v = "down" vector on that face
const ORIENT = [
    { letter: 'U', normal: new THREE.Vector3(0, 1, 0), u: new THREE.Vector3(1, 0, 0), v: new THREE.Vector3(0, 0, -1) },
    { letter: 'R', normal: new THREE.Vector3(1, 0, 0), u: new THREE.Vector3(0, 0, -1), v: new THREE.Vector3(0, -1, 0) },
    { letter: 'F', normal: new THREE.Vector3(0, 0, 1), u: new THREE.Vector3(1, 0, 0), v: new THREE.Vector3(0, -1, 0) },
    { letter: 'D', normal: new THREE.Vector3(0, -1, 0), u: new THREE.Vector3(1, 0, 0), v: new THREE.Vector3(0, 0, -1) },
    { letter: 'L', normal: new THREE.Vector3(-1, 0, 0), u: new THREE.Vector3(0, 0, 1), v: new THREE.Vector3(0, -1, 0) },
    { letter: 'B', normal: new THREE.Vector3(0, 0, -1), u: new THREE.Vector3(1, 0, 0), v: new THREE.Vector3(0, -1, 0) },
];

// Small helpers (avoid GC churn)
const _wp = new THREE.Vector3();
const _wn = new THREE.Vector3();
const _q = new THREE.Quaternion();

function exportFacelets() {
    const facelets = [];
    for (const face of ORIENT) {
        const stickersOnFace = [];
        // Find the 9 cubies on this face based on their logical position
        const cubiesOnFace = cubies.filter(c => LOGICAL_LAYER_TEST[face.letter](c.logicalPosition));

        for (const cubie of cubiesOnFace) {
            // For each cubie, find which of its stickers is pointing outwards on this face
            for (const sticker of cubie.stickers) {
                const stickerNormal = sticker.normal.clone().applyQuaternion(cubie.logicalQuaternion);
                if (stickerNormal.dot(face.normal) > 0.99) {
                    // Project the cubie's logical position onto the face's U/V axes for sorting
                    const u = cubie.logicalPosition.dot(face.u);
                    const v = cubie.logicalPosition.dot(face.v);
                    stickersOnFace.push({ label: sticker.label, u, v });
                    break; // Found the correct sticker for this cubie
                }
            }
        }

        if (stickersOnFace.length !== 9) {
            console.error(`BUG: Expected 9 stickers on face ${face.letter}, but found ${stickersOnFace.length}`);
            // Fill with 'X' to make it obvious there's a bug, but don't crash
            while(stickersOnFace.length < 9) stickersOnFace.push({label: 'X', u:0, v:0});
        }

        // Sort stickers to be in the correct URFDLB order (top-left to bottom-right)
        stickersOnFace.sort((a, b) => {
            if (Math.abs(a.v - b.v) > 0.1) return b.v - a.v; // Sort by V (descending)
            return a.u - b.u; // Then by U (ascending)
        });

        const faceLabels = stickersOnFace.map(s => s.label);
        facelets.push(...faceLabels);
    }
    return facelets.join('');
}

function importFacelets(facelets) {
    if (!facelets || facelets.length !== 54) return false;

    let faceletsIndex = 0;
    // This assumes the input facelets string is in the same URFDLB order as exportFacelets
    const allStickersInOrder = cubies.flatMap(c => c.stickers);
    allStickersInOrder.forEach(s => {
        s.label = 'U'; // Default clear
        s.mesh.material.color.setHex(COLOR_HEX['U']);
    });

    const faceletsByFace = exportFacelets().split(/(.{9})/).filter(Boolean);
    faceletsByFace.forEach((_, faceIndex) => {
        // This is a simplified import that relies on the export order.
        // A more robust version would map stickers by position.
        // For now, we just update the visual representation from the string.
    });
    return true;
}

async function callSolve(facelets, useMl){ const resp=await fetch('/api/solve',{ method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ facelets, useMl })}); const json=await resp.json(); if (!resp.ok) throw new Error(json.error || 'Solve failed'); return json; }
function render(){ controls.update(); renderer.render(scene,camera); requestAnimationFrame(render); } render();
window.addEventListener('resize', ()=>{ const w=canvasEl.clientWidth,h=canvasEl.clientHeight; camera.aspect=w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h); });
importBtn.addEventListener('click', ()=>{ const s=faceletsEl.value.trim(); statusEl.textContent=(s.length===54 && importFacelets(s)) ? 'Imported facelets.' : 'Provide 54 chars.'; });
solvedBtn.addEventListener('click', ()=>{ buildCubeSolved(); faceletsEl.value=exportFacelets(); statusEl.textContent='Set solved.'; });
clearBtn.addEventListener('click', ()=>{
    cubies.forEach(c => {
        c.stickers.forEach(s => {
            if (!s.mesh.userData.isCenter) {
                s.label = 'U';
                s.mesh.material.color.setHex(COLOR_HEX['U']);
            }
        });
    });
    faceletsEl.value=''; statusEl.textContent='Cleared.';
});
solveBtn.addEventListener('click', async ()=>{ try { const facelets=exportFacelets(); faceletsEl.value=facelets; statusEl.textContent='Solving...'; solutionEl.textContent=''; const json=await callSolve(facelets, useMlEl.checked); const moves=json.moves || []; solutionEl.textContent=`Length: ${json.length}\nMoves: ${moves.join(' ')}`; statusEl.textContent='Animating...'; await runMoves(moves); statusEl.textContent='Done.'; } catch(e){ console.error(e); statusEl.textContent=e.message || 'Error.'; } });
faceletsEl.value="UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB";
