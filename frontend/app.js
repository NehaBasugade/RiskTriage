
const API_URL = "/predict";
const STORAGE_KEY = "risktriage:last_inputs:v2";

document.getElementById("apiUrlLabel").textContent =
  window.location.origin + API_URL;


const FEATURE_DEFS = [


  {
  name: "1st Road Class",
  label: "Road type (importance)",
  type: "select",
  help: "Model expects numeric road class codes.",
  options: [
    { label: "Major road / highway (A road)", value: 1 },
    { label: "Secondary road (B road)", value: 2 },
    { label: "Local road (C road)", value: 3 },
    { label: "Residential / unclassified", value: 4 },
  ],
  },

  // ---- Date/time ----
  {
    name: "Accident Date",
    label: "Accident date",
    type: "date",
    help: "YYYY-MM-DD",
  },
  { name: "TimeHour", label: "Time of day (hour)", type: "number", min: 0, max: 23, step: 1, placeholder: "e.g., 14" },

  // ---- Person/case ----
  { name: "Age of Casualty", label: "Casualty age", type: "number", min: 0, max: 120, step: 1, placeholder: "e.g., 30" },
  {
    name: "Sex of Casualty",
    label: "Casualty sex",
    type: "select",
    options: ["Male", "Female", "Unknown"],
  },
  {
    name: "Casualty Class",
    label: "Casualty role",
    type: "select",
    options: ["Driver or rider", "Passenger", "Pedestrian", "Unknown"],
  },

  // ---- Environment ----
  {
    name: "Weather Conditions",
    label: "Weather",
    type: "select",
    options: ["Fine no high winds", "Raining no high winds", "Snowing", "Fog or mist", "Unknown"],
  },
  {
    name: "Lighting Conditions",
    label: "Lighting",
    type: "select",
    options: ["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit", "Unknown"],
  },
  {
    name: "Road Surface",
    label: "Road surface",
    type: "select",
    options: ["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood", "Unknown"],
  },

  // ---- Vehicles ----
  { name: "Number of Vehicles", label: "Vehicles involved", type: "number", min: 1, max: 50, step: 1, placeholder: "e.g., 2" },
  {
    name: "Type of Vehicle",
    label: "Primary vehicle type",
    type: "select",
    options: ["Car", "Motorcycle", "Bus or coach", "Goods vehicle", "Bicycle", "Other", "Unknown"],
  },

  // ---- Location (UI asks for meters
  {
    name: "GridBin_E",
    label: "Location Easting (meters)",
    type: "number",
    min: 0,
    max: 9999999,
    step: 1,
    placeholder: "e.g., 432145",
    //help: "Enter Easting in meters. UI converts to a privacy-preserving location zone (km bin).",
    transform: (v) => Math.floor(Number(v) / 1000),
  },
  {
    name: "GridBin_N",
    label: "Location Northing (meters)",
    type: "number",
    min: 0,
    max: 9999999,
    step: 1,
    placeholder: "e.g., 389221",
    //help: "Enter Northing in meters. UI converts to a privacy-preserving location zone (km bin).",
    transform: (v) => Math.floor(Number(v) / 1000),
  },
];


const EXAMPLE_INPUT = {
  "1st Road Class": 1,
  "Accident Date": "2020-01-15",
  "TimeHour": 18,
  "Age of Casualty": 29,
  "Sex of Casualty": "Male",
  "Casualty Class": "Driver or rider",
  "Weather Conditions": "Raining no high winds",
  "Lighting Conditions": "Darkness - lights lit",
  "Road Surface": "Wet or damp",
  "Number of Vehicles": 2,
  "Type of Vehicle": "Car",
  "GridBin_E": 432145, // meters -> will be binned
  "GridBin_N": 389221, // meters -> will be binned
};

// ====== DOM ======
const elFields = document.getElementById("formFields");
const elForm = document.getElementById("riskForm");
const elStatus = document.getElementById("status");

const elResultEmpty = document.getElementById("resultEmpty");
const elResultBox = document.getElementById("resultBox");
const elErrorBox = document.getElementById("errorBox");

const elTriagePill = document.getElementById("triagePill");
const elRiskScore = document.getElementById("riskScore");
const elMeaning = document.getElementById("meaning");
const elThresholds = document.getElementById("thresholds");

const elWhyJson = document.getElementById("whyJson");
const elWhyPanel = document.getElementById("whyPanel");

// Hero “latest prediction”
const elHeroPill = document.getElementById("heroPill");
const elHeroScore = document.getElementById("heroScore");
const elHeroHint = document.getElementById("heroHint");

//document.getElementById("apiUrlLabel").textContent = API_URL;
document.getElementById("apiUrlLabel").textContent =
  window.location.origin + API_URL;

// ====== Build form ======
function buildForm() {
  elFields.innerHTML = "";

  for (const f of FEATURE_DEFS) {
    const wrap = document.createElement("div");
    wrap.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", f.name);
    label.textContent = f.label;

    let input;

    if (f.type === "select") {
      input = document.createElement("select");
      input.id = f.name;
      input.name = f.name;

      for (const opt of f.options) {
        const o = document.createElement("option");
        if (typeof opt === "string") {
          o.value = opt;
          o.textContent = opt;
        } else {
          o.value = opt.value;
          o.textContent = opt.label;
        }
        input.appendChild(o);
      }
    } else if (f.type === "number") {
      input = document.createElement("input");
      input.type = "number";
      input.id = f.name;
      input.name = f.name;
      if (f.min !== undefined) input.min = String(f.min);
      if (f.max !== undefined) input.max = String(f.max);
      if (f.step !== undefined) input.step = String(f.step);
      input.placeholder = f.placeholder || "";
    } else if (f.type === "date") {
      input = document.createElement("input");
      input.type = "date";
      input.id = f.name;
      input.name = f.name;
    } else {
      input = document.createElement("input");
      input.type = "text";
      input.id = f.name;
      input.name = f.name;
      input.placeholder = f.placeholder || "";
    }

    input.addEventListener("change", persistInputs);
    input.addEventListener("input", persistInputs);

    wrap.appendChild(label);
    wrap.appendChild(input);

    if (f.help) {
      const help = document.createElement("div");
      help.className = "help";
      help.textContent = f.help;
      wrap.appendChild(help);
    }

    elFields.appendChild(wrap);
  }
}

function getInputs() {
  const payload = {};

  for (const f of FEATURE_DEFS) {
    const el = document.getElementById(f.name);
    if (!el) continue;

    if (f.type === "number") {
      const v = el.value.trim();
      if (v === "") {
        payload[f.name] = null;
      } else {
        // allow transform (e.g., meters -> km bin)
        const raw = Number(v);
        payload[f.name] = f.transform ? f.transform(raw) : raw;
      }
    } else {
      payload[f.name] = el.value;
    }
  }

  // Hard guard: API requires NO nulls for schema-only demos? (your backend allows None)
  // Keep as-is.

  return payload;
}

function setInputs(obj) {
  for (const f of FEATURE_DEFS) {
    const el = document.getElementById(f.name);
    if (!el) continue;

    if (obj[f.name] === undefined || obj[f.name] === null) {
      if (f.type === "number" || f.type === "date") el.value = "";
      continue;
    }
    el.value = String(obj[f.name]);
  }
  persistInputs();
}

function persistInputs() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(getInputsRawForStorage()));
  } catch (_) {}
}

function getInputsRawForStorage() {
  const raw = {};
  for (const f of FEATURE_DEFS) {
    const el = document.getElementById(f.name);
    if (!el) continue;

    if (f.type === "number") {
      const v = el.value.trim();
      raw[f.name] = v === "" ? null : Number(v);
    } else {
      raw[f.name] = el.value;
    }
  }
  return raw;
}

function loadPersistedInputs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const obj = JSON.parse(raw);
    setInputs(obj);
  } catch (_) {}
}

function resetInputs() {
  localStorage.removeItem(STORAGE_KEY);
  for (const f of FEATURE_DEFS) {
    const el = document.getElementById(f.name);
    if (!el) continue;
    if (f.type === "number" || f.type === "date") el.value = "";
    else if (f.type === "select") el.selectedIndex = 0;
    else el.value = "";
  }
  hideResult();
  hideError();
  elStatus.textContent = "";
  // reset hero
  elHeroPill.textContent = "—";
  elHeroPill.classList.remove("urgent", "review", "low");
  elHeroPill.classList.add("low");
  elHeroScore.textContent = "—";
  elHeroHint.textContent = "Run a prediction to populate this.";
}

function showError(msg) {
  elErrorBox.textContent = msg;
  elErrorBox.classList.remove("hidden");
}
function hideError() {
  elErrorBox.classList.add("hidden");
  elErrorBox.textContent = "";
}

function hideResult() {
  elResultEmpty.classList.remove("hidden");
  elResultBox.classList.add("hidden");
}

function updateHero({ risk_score, triage_level }) {
  const tl = String(triage_level || "").toUpperCase();
  const score = Number(risk_score);

  elHeroPill.textContent = triage_level ?? "—";
  elHeroPill.classList.remove("urgent", "review", "low");
  if (tl === "URGENT") elHeroPill.classList.add("urgent");
  else if (tl === "REVIEW") elHeroPill.classList.add("review");
  else elHeroPill.classList.add("low");

  elHeroScore.textContent = Number.isFinite(score) ? score.toFixed(3) : String(risk_score);

  if (tl === "URGENT") elHeroHint.textContent = "High-risk case — prioritize immediate review.";
  else if (tl === "REVIEW") elHeroHint.textContent = "Moderate risk — flag for review.";
  else if (tl === "LOW") elHeroHint.textContent = "Low priority — likely lower severity.";
  else elHeroHint.textContent = "Run a prediction to populate this.";
}

function showResult({ risk_score, triage_level, thresholds, why }) {
  elResultEmpty.classList.add("hidden");
  elResultBox.classList.remove("hidden");

  const score = Number(risk_score);
  elRiskScore.textContent = Number.isFinite(score) ? score.toFixed(3) : String(risk_score);

  // pill styling
  elTriagePill.textContent = triage_level ?? "—";
  elTriagePill.classList.remove("urgent", "review", "low");
  const tl = String(triage_level || "").toUpperCase();
  if (tl === "URGENT") elTriagePill.classList.add("urgent");
  else if (tl === "REVIEW") elTriagePill.classList.add("review");
  else if (tl === "LOW") elTriagePill.classList.add("low");

  // meaning
  if (tl === "URGENT") elMeaning.textContent = "High-risk case. Act now — prioritize immediate review.";
  else if (tl === "REVIEW") elMeaning.textContent = "Moderate risk. Flag for review and follow-up triage.";
  else elMeaning.textContent = "Low priority. Likely lower severity compared to other cases.";

  // thresholds
  if (thresholds && typeof thresholds === "object") {
    const u = thresholds.urgent ?? thresholds.urgent_threshold ?? thresholds.t_urgent;
    const r = thresholds.review ?? thresholds.review_threshold ?? thresholds.t_review;
    const parts = [];
    if (u !== undefined) parts.push(`URGENT≈${Number(u).toFixed(3)}`);
    if (r !== undefined) parts.push(`REVIEW≈${Number(r).toFixed(3)}`);
    elThresholds.textContent = parts.length ? parts.join(" • ") : "—";
  } else {
    elThresholds.textContent = "—";
  }

  // why panel
  if (why !== undefined) {
    elWhyPanel.open = false;
    elWhyJson.textContent = JSON.stringify(why, null, 2);
  } else {
    elWhyPanel.open = false;
    elWhyJson.textContent = "";
  }

  // hero update
  updateHero({ risk_score, triage_level });
}

// ====== API ======
async function predict(payload) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error (${res.status}): ${text}`);
  }
  return await res.json();
}

// ====== Events ======
elForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  elStatus.textContent = "Predicting…";

  const payload = getInputs();

  try {
    const out = await predict(payload);

    showResult({
      risk_score: out.risk_score,
      triage_level: out.triage_level,
      thresholds: out.thresholds ?? out.thresholds_used ?? out,
      why: out.why,
    });

    persistInputs();
    elStatus.textContent = "Done.";
  } catch (err) {
    showError(err.message || "Prediction failed.");
    elStatus.textContent = "";
  }
});

document.getElementById("btnLoadExample").addEventListener("click", () => {
  setInputs(EXAMPLE_INPUT);
  elStatus.textContent = "Example loaded.";
});

document.getElementById("btnReset").addEventListener("click", () => {
  resetInputs();
});

// ====== Init ======
buildForm();
loadPersistedInputs();
hideResult();