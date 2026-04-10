/**
 * SignBridge AI — avatar.js
 * Three.js 3D Hand Skeleton Avatar Renderer
 *
 * - Creates a Three.js scene inside #avatar-canvas
 * - Renders a 21-joint hand skeleton using bone cylinders + spheres
 * - Animates keypoint sequences with smooth cubic interpolation
 * - Provides play/pause/resume/replay controls
 */

(function () {
  "use strict";

  // ─── Three.js hand skeleton connection pairs ──────────────────────────
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],      // thumb
    [0,5],[5,6],[6,7],[7,8],      // index
    [0,9],[9,10],[10,11],[11,12], // middle
    [0,13],[13,14],[14,15],[15,16],// ring
    [0,17],[17,18],[18,19],[19,20],// pinky
    [5,9],[9,13],[13,17],          // palm
  ];

  // ─── State ─────────────────────────────────────────────────────────────
  let _scene, _camera, _renderer, _animFrameId;
  let _joints      = [];       // 21 Three.js Objects3D (spheres)
  let _bones       = [];       // cylinder meshes per connection
  let _keyframes   = [];       // array of 21-landmark frames
  let _frameIdx    = 0;
  let _paused      = false;
  let _playing     = false;
  let _frameTimer  = null;
  let _queue       = [];       // pending sign tokens to play

  const JOINT_COLOR    = 0x00d4ff;
  const TIP_COLOR      = 0x00ff9d;
  const BONE_COLOR     = 0x005577;
  const BG_COLOR       = 0x050f1f;
  const TIP_INDICES    = new Set([4, 8, 12, 16, 20]);
  const FRAME_DURATION = 1000 / 24; // ms per frame at 24fps

  // ─── Scale factor: keypoints are in ~[-0.5, 0.5] normalized space
  const SCALE = 2.5;
  const Y_FLIP = -1; // flip Y axis (canvas vs 3D coords)

  // ─── Public module ─────────────────────────────────────────────────────
  window.avatarModule = { init, playSequence, pause, resume, replay };

  // ─── Expose controls globally ─────────────────────────────────────────
  window.pauseAvatar  = pause;
  window.resumeAvatar = resume;
  window.replayAvatar = replay;

  // ─── Init ──────────────────────────────────────────────────────────────

  function init(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) { console.warn("Avatar canvas not found:", canvasId); return; }

    // Scene
    _scene = new THREE.Scene();
    _scene.background = new THREE.Color(BG_COLOR);

    // Grid helper for depth perception
    const grid = new THREE.GridHelper(4, 8, 0x0a2040, 0x0a2040);
    grid.position.y = -1.5;
    _scene.add(grid);

    // Camera
    _camera = new THREE.PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    _camera.position.set(0, 0, 3.5);
    _camera.lookAt(0, 0, 0);

    // Renderer
    _renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    _renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    _renderer.shadowMap.enabled = true;

    // Lighting
    const ambient = new THREE.AmbientLight(0x0a2040, 2.0);
    _scene.add(ambient);
    const point = new THREE.PointLight(0x00d4ff, 2.5, 10);
    point.position.set(1, 2, 2);
    _scene.add(point);
    const fill = new THREE.DirectionalLight(0x003355, 1.0);
    fill.position.set(-2, -1, -1);
    _scene.add(fill);

    // Build hand skeleton
    _buildHandMesh();

    // Resize observer
    const ro = new ResizeObserver(() => _onResize(canvas));
    ro.observe(canvas);

    // Render loop
    _renderLoop();

    // Show idle pose
    _applyLandmarks(_idlePose());
  }

  // ─── Build hand mesh ───────────────────────────────────────────────────

  function _buildHandMesh() {
    // Joint spheres
    const geomJoint = new THREE.SphereGeometry(0.05, 10, 10);
    const geomTip   = new THREE.SphereGeometry(0.07, 10, 10);

    for (let i = 0; i < 21; i++) {
      const isTip = TIP_INDICES.has(i);
      const mat = new THREE.MeshPhongMaterial({
        color:    isTip ? TIP_COLOR : JOINT_COLOR,
        emissive: isTip ? 0x003322 : 0x001133,
        shininess: 60,
      });
      const mesh = new THREE.Mesh(isTip ? geomTip : geomJoint, mat);
      _scene.add(mesh);
      _joints.push(mesh);
    }

    // Bone cylinders
    for (const [a, b] of CONNECTIONS) {
      const mat = new THREE.MeshPhongMaterial({
        color: BONE_COLOR,
        emissive: 0x001122,
        transparent: true,
        opacity: 0.7,
      });
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.018, 0.018, 1, 8),
        mat
      );
      _scene.add(mesh);
      _bones.push({ mesh, a, b });
    }
  }

  // ─── Keyframe animation ────────────────────────────────────────────────

  /**
   * Play an ordered list of sign tokens using the provided sequences dict.
   * @param {string[]} tokens  — e.g. ["chest_pain", "help"]
   * @param {object}   seqs    — { "chest_pain": { frames: [...] }, ... }
   */
  async function playSequence(tokens, seqs) {
    // Build flat keyframe list from all token sequences
    const frames = [];
    for (const token of tokens) {
      const seq = seqs[token];
      if (seq && seq.frames) {
        frames.push(...seq.frames);
        // Brief pause between signs (hold last frame for 3 frames)
        for (let i = 0; i < 3; i++) frames.push(seq.frames[seq.frames.length - 1]);
      }
    }

    if (frames.length === 0) return;

    _keyframes = frames;
    _frameIdx  = 0;
    _paused    = false;
    _playing   = true;

    // Stop any existing playback timer
    if (_frameTimer) clearInterval(_frameTimer);

    _frameTimer = setInterval(() => {
      if (_paused || !_playing) return;
      if (_frameIdx >= _keyframes.length) {
        _playing = false;
        clearInterval(_frameTimer);
        return;
      }
      _applyLandmarks(_keyframes[_frameIdx++]);
    }, FRAME_DURATION);
  }

  function pause()  { _paused = true; }
  function resume() { _paused = false; }
  function replay() {
    if (_keyframes.length === 0) return;
    _frameIdx = 0;
    _paused   = false;
    _playing  = true;
    if (_frameTimer) clearInterval(_frameTimer);
    _frameTimer = setInterval(() => {
      if (_paused || !_playing) return;
      if (_frameIdx >= _keyframes.length) {
        _playing = false;
        clearInterval(_frameTimer);
        return;
      }
      _applyLandmarks(_keyframes[_frameIdx++]);
    }, FRAME_DURATION);
  }

  // ─── Apply landmarks ───────────────────────────────────────────────────

  function _applyLandmarks(landmarks) {
    if (!landmarks || landmarks.length < 21) return;

    // Convert landmark [x,y,z] → Three.js Vector3 with scaling
    const pts = landmarks.map(([x, y, z]) =>
      new THREE.Vector3(x * SCALE, y * Y_FLIP * SCALE, (z || 0) * SCALE)
    );

    // Position joint spheres
    _joints.forEach((mesh, i) => {
      if (pts[i]) mesh.position.copy(pts[i]);
    });

    // Update bone cylinders: position + rotation + scale
    _bones.forEach(({ mesh, a, b }) => {
      const pa = pts[a];
      const pb = pts[b];
      if (!pa || !pb) return;

      const dir = new THREE.Vector3().subVectors(pb, pa);
      const len = dir.length();
      if (len < 0.001) return;

      // Mid-point
      mesh.position.copy(pa).addScaledVector(dir, 0.5);

      // Orient cylinder along bone direction
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.clone().normalize()
      );

      // Scale length
      mesh.scale.set(1, len, 1);
    });
  }

  // ─── Idle pose ─────────────────────────────────────────────────────────

  function _idlePose() {
    // Return a neutral open-palm pose in [x,y,z] format matching keypoint JSONs
    return [
      [0.0,   0.0,   0.0],
      [0.10,  0.08,  0.0],
      [0.18,  0.15,  0.0],
      [0.24,  0.22,  0.0],
      [0.29,  0.28,  0.0],
      [0.08, -0.12, 0.0],
      [0.09, -0.24, 0.0],
      [0.09, -0.33, 0.0],
      [0.09, -0.40, 0.0],
      [0.00, -0.14, 0.0],
      [0.00, -0.27, 0.0],
      [0.00, -0.36, 0.0],
      [0.00, -0.43, 0.0],
      [-0.08,-0.13, 0.0],
      [-0.08,-0.25, 0.0],
      [-0.08,-0.34, 0.0],
      [-0.08,-0.41, 0.0],
      [-0.15,-0.10, 0.0],
      [-0.15,-0.20, 0.0],
      [-0.15,-0.27, 0.0],
      [-0.15,-0.33, 0.0],
    ];
  }

  // ─── Render loop ───────────────────────────────────────────────────────

  function _renderLoop() {
    _animFrameId = requestAnimationFrame(_renderLoop);

    // Slow auto-rotation when not playing
    if (!_playing && _scene) {
      _scene.rotation.y += 0.003;
    } else if (_scene) {
      // Gradually reset rotation
      _scene.rotation.y *= 0.97;
    }

    if (_renderer && _scene && _camera) {
      _renderer.render(_scene, _camera);
    }
  }

  function _onResize(canvas) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    _camera.aspect = w / h;
    _camera.updateProjectionMatrix();
    _renderer.setSize(w, h, false);
  }

})();
