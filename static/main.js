import * as THREE from "three";
import * as GaussianSplats3D from "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const btnGenerate = document.getElementById("btnGenerate");
const statusBar = document.getElementById("statusBar");
const viewerHost = document.getElementById("viewerHost");
const viewerPlaceholder = document.getElementById("viewerPlaceholder");
const sceneSelect = document.getElementById("sceneSelect");
const btnFullscreen = document.getElementById("btnFullscreen");
const splatScale = document.getElementById("splatScale");
const limitSplatsCheck = document.getElementById("limitSplatsCheck");
const maxSplatsInput = document.getElementById("maxSplatsInput");
const exportSpzCheck = document.getElementById("exportSpzCheck");
const splatInfo = document.getElementById("splatInfo");

/** World-units per key press (fly mode); orbit mode scales slightly with distance. */
const DOLLY_BASE = 0.14;
/** Below this distance to the orbit target, dolly along camera forward and move the target too (no hard stop). */
const FLY_THRESHOLD = 0.22;

const _dollyDir = new THREE.Vector3();

let viewer = null;
let currentFile = null;

function setStatus(text, kind = "") {
  statusBar.textContent = text;
  statusBar.classList.remove("error", "working");
  if (kind) statusBar.classList.add(kind);
}

function formatSplatLine(count, full, limited) {
  const c = Number(count).toLocaleString();
  if (limited && full != null && Number(full) > Number(count)) {
    return `Splats: ${c} (capped from ${Number(full).toLocaleString()})`;
  }
  return `Splats: ${c}`;
}

function setSplatInfoFromApi(data) {
  if (!splatInfo || data.splat_count == null) return;
  let text = formatSplatLine(
    data.splat_count,
    data.splat_count_full,
    Boolean(data.splat_limit_applied)
  );
  if (data.decimate_error) {
    text += ` — ${data.decimate_error}`;
  }
  splatInfo.innerHTML = "";
  splatInfo.appendChild(document.createTextNode(text));
  if (data.spz_url) {
    const dl = document.createElement("a");
    dl.href = data.spz_url;
    dl.download = "";
    dl.textContent = " (download .spz)";
    dl.className = "spz-link";
    splatInfo.appendChild(dl);
  }
}

function setSplatInfoFromSelect() {
  if (!splatInfo) return;
  const opt = sceneSelect.selectedOptions[0];
  if (!opt || !opt.value) {
    splatInfo.textContent = "Splats: —";
    return;
  }
  const c = opt.dataset.splatCount;
  if (c === undefined || c === "") {
    splatInfo.textContent = "Splats: —";
    return;
  }
  let text = formatSplatLine(
    Number(c),
    opt.dataset.splatCountFull ? Number(opt.dataset.splatCountFull) : null,
    opt.dataset.splatLimited === "1"
  );
  if (opt.dataset.decimateError) {
    text += ` — ${opt.dataset.decimateError}`;
  }
  splatInfo.innerHTML = "";
  splatInfo.appendChild(document.createTextNode(text));
  if (opt.dataset.spzUrl) {
    const dl = document.createElement("a");
    dl.href = opt.dataset.spzUrl;
    dl.download = "";
    dl.textContent = " (download .spz)";
    dl.className = "spz-link";
    splatInfo.appendChild(dl);
  }
}

async function disposeViewer() {
  if (viewer) {
    try {
      await viewer.dispose();
    } catch (_) {
      /* ignore */
    }
    viewer = null;
  }
  viewerHost.innerHTML = "";
}

function showPlaceholder(show) {
  viewerPlaceholder.classList.toggle("hidden", !show);
}

function isFormControlTarget(target) {
  const el = /** @type {HTMLElement | null} */ (target);
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

/**
 * Move along view: orbit-dolly toward/away from target while far; when close, translate
 * camera + target on camera forward so movement does not freeze at the focal point.
 */
function dollyCameraAlongView(sign) {
  if (!viewer?.camera || !viewer.controls) return;
  const camera = viewer.camera;
  const controls = viewer.controls;

  const dist = camera.position.distanceTo(controls.target);
  _dollyDir.subVectors(controls.target, camera.position);

  const useFly = dist < FLY_THRESHOLD || _dollyDir.lengthSq() < 1e-12;
  if (useFly) {
    camera.getWorldDirection(_dollyDir);
    const step = sign * DOLLY_BASE;
    camera.position.addScaledVector(_dollyDir, step);
    controls.target.addScaledVector(_dollyDir, step);
  } else {
    _dollyDir.normalize();
    const step = sign * DOLLY_BASE * Math.min(1, Math.max(dist * 0.3, 0.2));
    camera.position.addScaledVector(_dollyDir, step);
  }

  controls.minDistance = 0;
  controls.maxDistance = Infinity;
  controls.update();
}

window.addEventListener(
  "keydown",
  (e) => {
    if (!viewer?.camera || !viewer.controls) return;
    if (isFormControlTarget(e.target)) return;

    const up = e.code === "ArrowUp";
    const down = e.code === "ArrowDown";
    const left = e.code === "ArrowLeft";
    const right = e.code === "ArrowRight";
    if (!up && !down && !left && !right) return;

    e.preventDefault();
    const forward = up || right;
    const back = down || left;
    const sign = forward && !back ? 1 : back && !forward ? -1 : 0;
    if (sign === 0) return;
    dollyCameraAlongView(sign);
  },
  { passive: false }
);

async function loadSplatUrl(url) {
  await disposeViewer();
  showPlaceholder(false);
  splatScale.disabled = true;

  const root = document.createElement("div");
  root.style.width = "100%";
  root.style.height = "100%";
  root.style.position = "absolute";
  root.style.inset = "0";
  viewerHost.appendChild(root);

  viewer = new GaussianSplats3D.Viewer({
    rootElement: root,
    cameraUp: [0, -1, -0.5],
    initialCameraPosition: [-0.5, -2.5, 4],
    initialCameraLookAt: [0, 0, 0.5],
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort: false,
    sphericalHarmonicsDegree: 0,
    logLevel: GaussianSplats3D.LogLevel.None,
  });

  const fullUrl = url.startsWith("http") ? url : `${window.location.origin}${url}`;

  try {
    await viewer.addSplatScene(fullUrl, {
      splatAlphaRemovalThreshold: 2,
      showLoadingUI: true,
      position: [0, 0, 0],
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
    });
    viewer.start();
    if (viewer.controls) {
      viewer.controls.minDistance = 0;
      viewer.controls.maxDistance = Infinity;
    }
  } catch (err) {
    await disposeViewer();
    showPlaceholder(true);
    splatScale.disabled = true;
    throw err;
  }

  splatScale.disabled = false;
  const scaleVal = parseFloat(splatScale.value);
  if (viewer.splatMesh && typeof viewer.splatMesh.setSplatScale === "function") {
    viewer.splatMesh.setSplatScale(scaleVal);
  }
}

splatScale.addEventListener("input", () => {
  const v = parseFloat(splatScale.value);
  if (viewer?.splatMesh?.setSplatScale) {
    viewer.splatMesh.setSplatScale(v);
  }
});

function setPreviewFile(file) {
  currentFile = file;
  if (!file) {
    previewImg.classList.add("hidden");
    dropZone.classList.remove("has-preview");
    previewImg.removeAttribute("src");
    btnGenerate.disabled = true;
    return;
  }
  const r = new FileReader();
  r.onload = () => {
    previewImg.src = /** @type {string} */ (r.result);
    previewImg.classList.remove("hidden");
    dropZone.classList.add("has-preview");
  };
  r.readAsDataURL(file);
  btnGenerate.disabled = false;
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) setPreviewFile(f);
  else if (f && /\.(heic|heif)$/i.test(f.name)) setPreviewFile(f);
});

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  setPreviewFile(f || null);
});

function syncMaxSplatsInputDisabled() {
  if (!limitSplatsCheck || !maxSplatsInput) return;
  maxSplatsInput.disabled = !limitSplatsCheck.checked;
}

limitSplatsCheck.addEventListener("change", syncMaxSplatsInputDisabled);
syncMaxSplatsInputDisabled();

btnGenerate.addEventListener("click", async () => {
  if (!currentFile) return;
  const fd = new FormData();
  fd.append("file", currentFile, currentFile.name);
  if (limitSplatsCheck.checked) {
    fd.append("limit_splats", "1");
    const raw = parseInt(maxSplatsInput.value, 10);
    const n = Number.isFinite(raw) && raw >= 1 ? Math.min(raw, 10_000_000) : 500_000;
    fd.append("max_splats", String(n));
  }
  fd.append("export_spz", exportSpzCheck.checked ? "1" : "0");
  setStatus("Generating…", "working");
  btnGenerate.disabled = true;
  try {
    const res = await fetch("/api/generate", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus(data.error || `Error ${res.status}`, "error");
      btnGenerate.disabled = false;
      return;
    }
    setStatus("Ready");
    setSplatInfoFromApi(data);
    await loadSplatUrl(data.ply_url);
    await refreshScenes(data.id);
    btnGenerate.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Network error", "error");
    btnGenerate.disabled = false;
  }
});

sceneSelect.addEventListener("change", async () => {
  const id = sceneSelect.value;
  if (!id) {
    if (splatInfo) splatInfo.textContent = "Splats: —";
    return;
  }
  setSplatInfoFromSelect();
  setStatus("Loading scene…", "working");
  try {
    await loadSplatUrl(`/api/scenes/${id}/splat.ply`);
    setStatus("Ready");
    setSplatInfoFromSelect();
  } catch (err) {
    console.error(err);
    setStatus("Failed to load scene", "error");
  }
});

btnFullscreen.addEventListener("click", () => {
  const el = document.querySelector(".viewer-shell");
  if (!el) return;
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    el.requestFullscreen().catch(() => {});
  }
});

async function refreshScenes(selectId = null) {
  try {
    const res = await fetch("/api/scenes");
    const list = await res.json();
    sceneSelect.innerHTML = '<option value="">Previous scenes…</option>';
    for (const s of list) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.label || s.id;
      if (s.splat_count != null) {
        opt.dataset.splatCount = String(s.splat_count);
        if (s.splat_count_full != null) opt.dataset.splatCountFull = String(s.splat_count_full);
        if (s.splat_limit_applied) opt.dataset.splatLimited = "1";
        if (s.decimate_error) opt.dataset.decimateError = s.decimate_error;
      }
      if (s.spz_url) opt.dataset.spzUrl = s.spz_url;
      sceneSelect.appendChild(opt);
    }
    if (selectId) {
      sceneSelect.value = selectId;
    }
  } catch (_) {
    /* ignore */
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    if (!h.ml_sharp_present) {
      setStatus("ml-sharp path missing — see server logs", "error");
    }
  } catch (_) {
    setStatus("Cannot reach API", "error");
  }
}

checkHealth();
refreshScenes();
