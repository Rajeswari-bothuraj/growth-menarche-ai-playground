
import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, ScatterChart, Scatter, ZAxis
} from "recharts";
import {
  BrainCircuit, Database, Play, Upload, Activity,
  SlidersHorizontal, BarChart3, Eye, GitBranch,
  Network, RefreshCw
} from "lucide-react";
import "./style.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed: ${response.status}`);
  }
  return data;
}

function Panel({ title, icon: Icon, children }) {
  return (
    <section className="panel">
      <h2>{Icon && <Icon size={17} />} {title}</h2>
      {children}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <small>{label}</small>
      <b>{value ?? "—"}</b>
    </div>
  );
}

function DecisionBoundaryPlot({ boundary, featureX, featureY }) {
  const width = 900;
  const height = 440;
  const pad = { left: 76, right: 24, top: 22, bottom: 64 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xs = boundary.x || [];
  const ys = boundary.y || [];
  if (!xs.length || !ys.length) return <div className="empty">No decision-boundary grid returned.</div>;

  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xStep = xs.length > 1 ? (xs[1] - xs[0]) : 1;
  const yStep = ys.length > 1 ? (ys[1] - ys[0]) : 1;
  const xScale = value => pad.left + ((value - xMin) / (xMax - xMin || 1)) * plotW;
  const yScale = value => pad.top + plotH - ((value - yMin) / (yMax - yMin || 1)) * plotH;
  const colors = ["#60a5fa", "#f59e0b", "#34d399", "#c084fc", "#f87171", "#22d3ee"];
  const colorFor = cls => colors[Number(cls) % colors.length];
  const cells = [];
  (boundary.predicted_class || []).forEach((row, yi) => {
    row.forEach((cls, xi) => {
      const cx = xScale(xs[xi]);
      const cy = yScale(ys[yi]);
      const cellW = Math.max(1, Math.abs(xScale(xs[xi] + xStep) - cx));
      const cellH = Math.max(1, Math.abs(yScale(ys[yi] + yStep) - cy));
      cells.push(
        <rect key={`cell-${yi}-${xi}`} x={cx - cellW / 2} y={cy - cellH / 2}
          width={cellW + 0.6} height={cellH + 0.6} fill={colorFor(cls)} opacity="0.28" />
      );
    });
  });
  const observations = (boundary.observations || []).filter(point =>
    Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y))
  );

  return (
    <div className="chart" style={{ height: 440 }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img"
        aria-label={`Predicted decision regions and observed records for ${featureX} and ${featureY}`}
        style={{ width: "100%", height: "100%" }}>
        <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="#ffffff" stroke="#cbd5e1" />
        {cells}
        {[0, 1, 2, 3, 4].map(i => {
          const x = pad.left + (plotW * i) / 4;
          const y = pad.top + (plotH * i) / 4;
          return <g key={`grid-${i}`}>
            <line x1={x} y1={pad.top} x2={x} y2={pad.top + plotH} stroke="#94a3b8" opacity="0.25" />
            <line x1={pad.left} y1={y} x2={pad.left + plotW} y2={y} stroke="#94a3b8" opacity="0.25" />
          </g>;
        })}
        {observations.map((point, i) => (
          <circle key={`obs-${i}`} cx={xScale(Number(point.x))} cy={yScale(Number(point.y))}
            r="4.2" fill={colorFor(point.class)} stroke="#172554" strokeWidth="1.1" opacity="0.95">
            <title>{`${point.label ?? boundary.class_labels?.[point.class] ?? `Class ${point.class}`} · ${featureX}: ${point.x} · ${featureY}: ${point.y}`}</title>
          </circle>
        ))}
        <text x={pad.left + plotW / 2} y={height - 14} textAnchor="middle" fontSize="14" fill="#334155">{featureX}</text>
        <text x="18" y={pad.top + plotH / 2} textAnchor="middle" fontSize="14" fill="#334155"
          transform={`rotate(-90 18 ${pad.top + plotH / 2})`}>{featureY}</text>
        <text x={pad.left} y={height - 42} fontSize="11" fill="#64748b">{xMin.toFixed(2)}</text>
        <text x={pad.left + plotW} y={height - 42} textAnchor="end" fontSize="11" fill="#64748b">{xMax.toFixed(2)}</text>
        <text x={pad.left - 10} y={pad.top + plotH} textAnchor="end" fontSize="11" fill="#64748b">{yMin.toFixed(2)}</text>
        <text x={pad.left - 10} y={pad.top + 5} textAnchor="end" fontSize="11" fill="#64748b">{yMax.toFixed(2)}</text>
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, padding: "4px 12px", fontSize: 12 }}>
        {(boundary.class_labels || []).map((label, index) => (
          <span key={`legend-${index}`} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <i style={{ width: 11, height: 11, borderRadius: "50%", background: colorFor(index), display: "inline-block" }} />
            {label}
          </span>
        ))}
        <span style={{ color: "#64748b" }}>Shaded area: model prediction · Circles: observed dataset records</span>
      </div>
    </div>
  );
}

function NetworkVisualization({
  network, featureNames, classLabels, activations, onWeightClick
}) {
  const [selectedWeight, setSelectedWeight] = useState(null);

  if (!network) {
    return <div className="empty">Train a model to inspect its network.</div>;
  }

  const parameters = network.parameters || {};
  const sizes = [
    network.input_size,
    ...(network.hidden_layers || []),
    network.output_size
  ];

  const weightKeys = Object.keys(parameters)
    .filter(k => /^network\.\d+\.weight$/.test(k))
    .sort((a, b) => Number(a.split(".")[1]) - Number(b.split(".")[1]));

  const width = Math.max(760, sizes.length * 210);
  const height = 390;
  const marginX = 90;
  const marginY = 55;
  const maxNodes = 12;
  const visible = sizes.map(n => Math.min(n, maxNodes));
  const gapX = (width - 2 * marginX) / Math.max(1, sizes.length - 1);

  const positions = visible.map((count, layer) => {
    const gapY = (height - 2 * marginY) / Math.max(1, count - 1);
    return Array.from({ length: count }, (_, node) => ({
      x: marginX + layer * gapX,
      y: count === 1 ? height / 2 : marginY + node * gapY
    }));
  });

  const connections = [];
  weightKeys.forEach((key, layer) => {
    const matrix = parameters[key] || [];
    (positions[layer + 1] || []).forEach((target, ti) => {
      (positions[layer] || []).forEach((source, si) => {
        const weight = matrix[ti]?.[si];
        if (typeof weight === "number") {
          connections.push({
            key: `${layer}-${si}-${ti}`,
            layer, source: si, target: ti,
            x1: source.x, y1: source.y,
            x2: target.x, y2: target.y,
            weight
          });
        }
      });
    });
  });

  function neuronName(layer, index) {
    if (layer === 0) return featureNames[index] || `Input ${index + 1}`;
    if (layer === sizes.length - 1) return classLabels[index] || `Class ${index + 1}`;
    return `Neuron ${index + 1}`;
  }

  function activationValue(layer, index) {
    if (!activations) return null;
    const item = activations.trace?.find(
      t => t.layer === `Hidden ${layer}`
    );
    const values = item?.activation?.[0];
    return values?.[index];
  }

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <svg width="100%" viewBox={`0 0 ${width} ${height}`}
          style={{ minWidth: 650 }}>
          {positions.map((_, i) => (
            <text key={`heading-${i}`}
              x={marginX + i * gapX} y="25"
              textAnchor="middle" fill="currentColor"
              fontSize="13" fontWeight="bold">
              {i === 0 ? "INPUT" :
                i === sizes.length - 1 ? "OUTPUT" : `HIDDEN ${i}`}
            </text>
          ))}

          {connections.map(c => (
            <line key={c.key}
              x1={c.x1} y1={c.y1} x2={c.x2} y2={c.y2}
              stroke={c.weight >= 0 ? "#16a34a" : "#ef4444"}
              strokeWidth={0.5 + Math.min(Math.abs(c.weight), 1) * 3}
              opacity={0.3 + Math.min(Math.abs(c.weight), 1) * 0.65}
              style={{ cursor: "pointer" }}
              onClick={() => {
                setSelectedWeight(c);
                onWeightClick?.(c);
              }}>
              <title>Click to inspect weight: {c.weight.toFixed(7)}</title>
            </line>
          ))}

          {positions.map((layer, li) =>
            layer.map((node, ni) => {
              const value = li > 0 && li < sizes.length - 1
                ? activationValue(li, ni) : null;
              return (
                <g key={`${li}-${ni}`}>
                  <circle cx={node.x} cy={node.y} r="13"
                    fill={li === 0 || li === sizes.length - 1
                      ? "#2563eb" : "#7c3aed"}
                    stroke="white" strokeWidth="2" />
                  <text x={node.x} y={node.y + 27}
                    textAnchor="middle" fontSize="9" fill="currentColor">
                    {neuronName(li, ni).slice(0, 13)}
                  </text>
                  {value !== null && value !== undefined && (
                    <text x={node.x} y={node.y - 19}
                      textAnchor="middle" fontSize="9" fill="#16834b">
                      {Number(value).toFixed(3)}
                    </text>
                  )}
                </g>
              );
            })
          )}
        </svg>
      </div>

      <p className="muted tiny">
        Click any connection to inspect its learned weight.
        Green = positive; red = negative. Activation values appear
        after running forward propagation.
        {sizes.some(n => n > maxNodes) &&
          " Only the first 12 neurons in each layer are drawn."}
      </p>

      {selectedWeight && (
        <div className="prediction">
          <b>Selected connection</b>
          <p>
            Layer {selectedWeight.layer + 1}, neuron {selectedWeight.source + 1}
            {" → "}
            neuron {selectedWeight.target + 1}
          </p>
          <b>Actual weight: {selectedWeight.weight.toFixed(8)}</b>
        </div>
      )}

      <div className="metrics">
        {sizes.map((n, i) => (
          <Metric key={i}
            label={i === 0 ? "Input neurons" :
              i === sizes.length - 1 ? "Output neurons" : `Hidden layer ${i}`}
            value={n} />
        ))}
      </div>
      <p className="muted tiny">
        Activation: {network.activation} · Engine: {network.engine}
      </p>
    </div>
  );
}


// Raw survey values are converted to the same standardized numeric values
// used when the DS3 CSV was originally prepared.
const DS3_NUMERIC_SCALING = {
  "num__EW6": { mean: 36.33729726994408, std: 9.866431416369313 },
  "num__ED6": { mean: 5.181818181818182, std: 4.920611696300128 }
};

function menarcheClassLabel(value) {
  const classValue = Number(value);
  if (classValue === 1) return "1 — Early menarche";
  if (classValue === 0) return "0 — Not early menarche";
  return `Class ${value}`;
}

function formatClassLabel(targetName, value) {
  const numeric = Number(value);
  if (targetName === "early_menarche") return menarcheClassLabel(value);
  if (targetName === "attended_school") {
    if (numeric === 1) return "1 — Attended school";
    if (numeric === 0) return "0 — Never attended school";
  }
  if (targetName === "child_is_girl") {
    if (numeric === 1) return "1 — Girl";
    if (numeric === 0) return "0 — Boy";
  }
  return String(value);
}


// DS4 raw survey values are converted to the same standardized scale used
// in the prepared DS4 CSV before the trained pipeline applies its own scaler.
const DS4_RAW_SCALING = {
  "num__BH6A": { mean: 17.50972471573908, std: 8.522021433983014 },
  "num__BH6B": { mean: 43.17456896551724, std: 41.84402590830027 },
  "num__BH5A": { mean: 6.5019762845849804, std: 3.279807282659779 },
  "num__BH5B": { mean: 24.991525423728813, std: 41.47257270223398 }
};

function prepareUserFeatureValue(column, rawValue) {
  const numericValue = Number(rawValue);
  if (!Number.isFinite(numericValue)) return rawValue;
  const scaling = DS4_RAW_SCALING[column];
  if (!scaling) return numericValue;
  // IHDS-II stores BH5B as a two-digit year code; accept a four-digit year
  // in the UI and convert it to the survey's stored two-digit code.
  const surveyValue = column === "num__BH5B" && numericValue >= 1900
    ? numericValue % 100 : numericValue;
  return (surveyValue - scaling.mean) / scaling.std;
}

const RELIGION_LABELS = {
  "1": "Hindu", "2": "Muslim", "3": "Christian", "4": "Sikh",
  "5": "Buddhist", "6": "Jain", "7": "Tribal", "8": "Other religion", "9": "No religion"
};
const CASTE_LABELS = {
  "1": "Brahmin", "2": "Forward / General", "3": "Other Backward Classes (OBC)",
  "4": "Scheduled Castes (SC)", "5": "Scheduled Tribes (ST)", "6": "Other social group"
};
const STATE_LABELS = {
  "1": "Jammu & Kashmir", "2": "Himachal Pradesh", "3": "Punjab", "4": "Chandigarh",
  "5": "Uttarakhand", "6": "Haryana", "7": "Delhi", "8": "Rajasthan",
  "9": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
  "13": "Nagaland", "14": "Manipur", "15": "Mizoram", "16": "Tripura",
  "17": "Meghalaya", "18": "Assam", "19": "West Bengal", "20": "Jharkhand",
  "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
  "25": "Daman & Diu", "26": "Dadra & Nagar Haveli", "27": "Maharashtra",
  "28": "Andhra Pradesh", "29": "Karnataka", "30": "Goa", "32": "Kerala",
  "33": "Tamil Nadu", "34": "Puducherry"
};

function categoryLabel(prefix, code) {
  const key = String(Number.parseFloat(code));
  if (prefix === "cat__ID11_") return RELIGION_LABELS[key] || `Religion category ${code}`;
  if (prefix === "cat__ID13_") return CASTE_LABELS[key] || `Social group ${code}`;
  if (prefix.startsWith("cat__STATEID_")) return STATE_LABELS[key] || `State category ${code}`;
  if (prefix === "cat__RO3_") return key === "1" ? "Male" : key === "2" ? "Female" : `Sex code ${code}`;
  if (prefix === "cat__ED4_") return key === "0" ? "No" : key === "1" ? "Yes" : `School attendance code ${code}`;
  return `Code ${code}`;
}

function displayFeatureName(column) {
  if (column === "num__RO5" || column === "RO5" || column.endsWith("__RO5")) return "Household member's age (years) — RO5";
  if (column === "cat__RO3_1.0" || column === "cat__RO3_1") return "Sex of household member — Male";
  if (column === "cat__RO3_2.0" || column === "cat__RO3_2") return "Sex of household member — Female";
  if (column === "current_age_years") return "Child's current age (years)";
  if (column === "num__EW6") return "Child's current age (years) — EW6";
  if (column === "num__ED6") return "Completed education (years) — ED6";
  if (column === "cat__ED4_0.0") return "Ever attended school — No";
  if (column === "cat__ED4_1.0") return "Ever attended school — Yes";
  if (column.startsWith("cat__ID11_")) return `Religion — ${categoryLabel("cat__ID11_", column.slice("cat__ID11_".length))}`;
  if (column.startsWith("cat__ID13_")) return `Caste / social group — ${categoryLabel("cat__ID13_", column.slice("cat__ID13_".length))}`;
  if (column.startsWith("cat__STATEID_")) return `Respondent's state — ${categoryLabel("cat__STATEID_", column.slice("cat__STATEID_".length))}`;
  if (column === "num__BH6A") return "Child's age at interview (completed years) — BH6A";
  if (column === "num__BH6B") return "Child's additional age (months) — BH6B";
  if (column === "num__BH5A") return "Child's birth month — BH5A";
  if (column === "num__BH5B") return "Child's birth year — BH5B";
  if (column === "child_is_girl") return "Child's sex — girl indicator";
  if (column === "attended_school") return "Ever attended school";
  if (column === "early_menarche") return "Early menarche (prediction target)";
  return column.replace(/^num__/, "").replace(/^cat__/, "").replaceAll("_", " ");
}

function featureHint(column) {
  if (column === "current_age_years") return "Enter the child's current age from 5 to 10 years.";
  if (column === "num__RO5") return "RO5 is stored as a standardized numeric value in this prepared dataset. Use the prepared-data scale for model testing; this is not age at first menstruation.";
  if (column === "num__EW6") return "Enter the child's current age (5–10 years). The app converts this value to the prepared model scale.";
  if (column === "num__ED6") return "Enter completed years of education (0–16). The app converts this value to the prepared DS3 model scale automatically.";
  if (column === "num__BH6A") return "This column is already standardized in the prepared dataset. Use the prepared-data numeric scale; it is not age at first menarche.";
  if (column === "num__BH6B") return "This column is already standardized in the prepared dataset. Use the prepared-data numeric scale; it is not menarche age.";
  if (column === "num__BH5A") return "This column is already standardized in the prepared dataset. Use the prepared-data numeric scale.";
  if (column === "num__BH5B") return "This column is already standardized in the prepared dataset. Use the prepared-data numeric scale.";
  if (column.startsWith("cat__ID11_")) return "Religion category recorded in the IHDS-II survey.";
  if (column.startsWith("cat__ID13_")) return "Caste / social-group category recorded in the IHDS-II survey.";
  if (column.startsWith("cat__STATEID_")) return "State category recorded in the IHDS-II survey.";
  if (column.startsWith("cat__RO3_")) return "Sex category recorded for this household member.";
  if (column.startsWith("cat__ED4_")) return "Whether the respondent ever attended school.";
  if (column === "child_is_girl") return "1 = girl; 0 = boy, according to the prepared dataset.";
  return "Enter the value in the same scale and coding used by the prepared dataset.";
}

function getOneHotCodes(featureNames, prefix) {
  const normalizedPrefix = prefix.toLowerCase();
  return [...new Set(featureNames
    .filter(name => String(name).toLowerCase().startsWith(normalizedPrefix))
    .map(name => String(name).slice(prefix.length).replace(/\.0$/, "")))]
    .sort((a, b) => Number(a) - Number(b));
}

function buildMenarcheValues(form, featureNames) {
  const values = Object.fromEntries(featureNames.map(name => [name, 0]));
  const age = Number(form.currentAge);
  const educationYears = Number(form.educationYears);

  if (!Number.isFinite(age) || age < 5 || age > 10) {
    throw new Error("Enter current age between 5 and 10 years.");
  }
  if (!Number.isFinite(educationYears) || educationYears < 0 || educationYears > 16) {
    throw new Error("Enter completed education between 0 and 16 years.");
  }

  for (const [column, rawValue] of [
    ["num__EW6", age],
    ["num__ED6", educationYears]
  ]) {
    const scaling = DS3_NUMERIC_SCALING[column];
    if (!featureNames.includes(column) || !scaling) {
      throw new Error(`The trained model is missing ${column}. Please select all 52 DS3 features and retrain.`);
    }
    values[column] = (rawValue - scaling.mean) / scaling.std;
  }

  const groups = [
    { prefix: "cat__ED4_", value: form.attendedSchool, label: "school attendance" },
    { prefix: "cat__ID11_", value: form.religion, label: "religion" },
    { prefix: "cat__ID13_", value: form.casteGroup, label: "caste group" },
    { prefix: "cat__STATEID_", value: form.state, label: "state" }
  ];

  for (const group of groups) {
    const options = getOneHotCodes(featureNames, group.prefix);
    // A category group is optional unless its columns were selected.
    if (!options.length) continue;
    if (!group.value || !options.includes(group.value)) {
      throw new Error(`Select ${group.label} from the categories included in your selected features.`);
    }
    const matchingColumns = featureNames.filter(name => name.startsWith(group.prefix));
    matchingColumns.forEach(column => {
      const code = column.slice(group.prefix.length).replace(/\.0$/, "");
      values[column] = code === String(group.value).replace(/\.0$/, "") ? 1 : 0;
    });
  }

  return values;
}

function MenarcheFields({ value, onChange, featureNames, freezeCategories = false, error = "" }) {
  const update = (key, next) => onChange(old => ({ ...old, [key]: next }));
  const selectField = (label, key, prefix, yesNo = false) => {
    const codes = getOneHotCodes(featureNames, prefix);
    if (!codes.length) return null;
    return (
      <label className="predInput">
        {label}
        <select value={value[key]} disabled={freezeCategories && ["attendedSchool", "religion", "casteGroup", "state"].includes(key)} onChange={e => update(key, e.target.value)}>
          <option value="">{codes.length ? `Select ${label.toLowerCase()}` : `No ${label.toLowerCase()} columns in selected model`}</option>
          {codes.map(code => <option key={code} value={code}>{categoryLabel(prefix, code)}</option>)}
        </select>
        {!codes.length && <small>Select all corresponding one-hot columns in the feature list and retrain to enable this dropdown.</small>}
      </label>
    );
  };

  return (
    <div className="formGrid">
      <p className="muted">
        Enter the respondent's survey details. For religion, caste/social group and state,
        choose the code recorded in the IHDS survey; do not guess a code.
      </p>
      <label className="predInput">
        Child's current age (years)
        <input type="number" min="5" max="10" step="1"
          value={value.currentAge}
          onChange={e => update("currentAge", e.target.value)}
          placeholder="e.g. 7"/>
        <small>Enter the child's current age from 5 to 10 years. This is not age at first menarche.</small>
        {error.toLowerCase().includes("current age") && <small style={{ color: "#b42318", fontWeight: 600 }}>{error}</small>}
      </label>
      <label className="predInput">
        Completed education (years)
        <input type="number" min="0" max="16" step="1"
          value={value.educationYears}
          onChange={e => update("educationYears", e.target.value)}
          placeholder="Type completed years, e.g. 10"/>
        <small>Enter a whole number from 0 to 16.</small>
      </label>
      {selectField("Ever attended school", "attendedSchool", "cat__ED4_", true)}
      {selectField("Religion (choose the respondent's survey code)", "religion", "cat__ID11_")}
      {selectField("Caste / social group (choose survey code)", "casteGroup", "cat__ID13_")}
      {selectField("State (choose survey code)", "state", "cat__STATEID_")}
    </div>
  );
}

const ONE_HOT_GROUPS = [
  { prefix: "cat__RO3_", label: "Household member's sex" },
  { prefix: "cat__ID11_", label: "Religion" },
  { prefix: "cat__ID13_", label: "Caste / social group" },
  { prefix: "cat__STATEID_", label: "Respondent's state" },
  { prefix: "cat__ED4_", label: "Ever attended school" }
];

function FeatureInputFields({ featureNames, values, onChange, disabledCategories = false, mode = "prediction", error = "" }) {
  const grouped = new Set();
  const groups = ONE_HOT_GROUPS.map(group => ({
    ...group,
    columns: featureNames.filter(name => name.startsWith(group.prefix))
  })).filter(group => group.columns.length);
  groups.forEach(group => group.columns.forEach(column => grouped.add(column)));
  const updateGroup = (group, chosen) => {
    onChange(old => ({
      ...old,
      ...Object.fromEntries(group.columns.map(column => [column, column === chosen ? "1" : "0"]))
    }));
  };
  const otherFeatures = featureNames.filter(name => !grouped.has(name));
  return (
    <div className="formGrid">
      {groups.map(group => {
        const selectedColumn = group.columns.find(column => Number(values[column]) === 1) || "";
        return (
          <label className="predInput" key={group.prefix}>
            {group.label}
            <select value={selectedColumn}
              disabled={disabledCategories && Boolean(selectedColumn)}
              onChange={event => updateGroup(group, event.target.value)}>
              <option value="">Choose {group.label.toLowerCase()}</option>
              {group.columns.map(column => {
                const code = column.slice(group.prefix.length);
                return <option key={column} value={column}>{categoryLabel(group.prefix, code)}</option>;
              })}
            </select>
            <small>Choose the category recorded in the IHDS-II survey. This selection is converted into the model's 0/1 encoded columns.</small>
          </label>
        );
      })}
      {otherFeatures.map(column => (
        <label className="predInput" key={column}>
          {displayFeatureName(column)}
          {column === "child_is_girl" || column === "attended_school" ? (
            <select value={values[column] ?? ""} onChange={event => onChange(old => ({ ...old, [column]: event.target.value }))}>
              <option value="">Choose an option</option>
              {column === "child_is_girl" ? <><option value="1">Girl</option><option value="0">Boy</option></> : <><option value="1">Yes</option><option value="0">No</option></>}
            </select>
          ) : (
            <input
              type="number"
              min={column === "current_age_years" || column === "num__BH6A" ? 5 : column === "num__BH5A" ? 1 : column === "num__BH6B" ? 0 : column === "num__BH5B" ? 1900 : undefined}
              max={column === "current_age_years" || column === "num__BH6A" ? 10 : column === "num__BH5A" ? 12 : column === "num__BH6B" ? 11 : column === "num__BH5B" ? 2026 : undefined}
              step={1}
              inputMode="numeric"
              value={values[column] ?? ""}
              placeholder={column === "current_age_years" || column === "num__BH6A" ? "5–10 years" : column === "num__BH5B" ? "e.g. 2018" : column === "num__BH5A" ? "1–12" : column === "num__BH6B" ? "0–11" : `Type ${displayFeatureName(column)}`}
              onChange={event => onChange(old => ({ ...old, [column]: event.target.value }))}/>
          )}
          <small>{featureHint(column)}</small>
          {((column === "num__BH6A" && error.toLowerCase().includes("completed age")) || (column === "current_age_years" && error.toLowerCase().includes("current age"))) && (
            <small style={{ color: "#b42318", fontWeight: 600 }}>{error}</small>
          )}
        </label>
      ))}
    </div>
  );
}

function App() {
  const [sets, setSets] = useState([]);
  const [slot, setSlot] = useState(1);
  const [selected, setSelected] = useState("");
  const [columns, setColumns] = useState([]);
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState([]);

  const [result, setResult] = useState(null);
  const [network, setNetwork] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [forward, setForward] = useState(null);
  const [boundary, setBoundary] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [weightInfo, setWeightInfo] = useState(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [layers, setLayers] = useState("8,4");
  const [activation, setActivation] = useState("relu");
  const [alpha, setAlpha] = useState(0.001);
  const [lr, setLr] = useState(0.001);
  const [epochs, setEpochs] = useState(100);
  const [testSize, setTestSize] = useState(0.2);

  const [forwardValues, setForwardValues] = useState({});
  const [predictionValues, setPredictionValues] = useState({});
  // Keep the exact raw feature values submitted to Predict for the forward-pass request.
  const [lastPredictionValues, setLastPredictionValues] = useState(null);
  const [menarcheInputs, setMenarcheInputs] = useState({
    currentAge: "", educationYears: "", attendedSchool: "",
    religion: "", casteGroup: "", state: ""
  });
  const [featureX, setFeatureX] = useState("");
  const [featureY, setFeatureY] = useState("");

  const current = sets.find(s => s.dataset_id === selected);
  // Dataset 2 contains the real derived outcome plus age/metadata columns.
  // Age at menarche and source metadata must never be model inputs or targets.
  const hiddenSyntheticColumns = new Set([
    "menarche_age_years", "age_source",
    "synthetic_menarche_age_years", "synthetic_early_menarche",
    "synthetic_test_record_id", "test_data_only", "age_assignment_method",
    "data_origin", "MH1E"
  ]);
  const availableColumns = useMemo(() => {
    if (current?.slot === 2 || current?.slot === 3) {
      return columns.filter(column => !hiddenSyntheticColumns.has(column) && !(current.slot === 2 && column === "current_age_years"));
    }
    return columns;
  }, [columns, current?.slot]);
  const targetOptions = (current?.slot === 2 || current?.slot === 3)
    ? availableColumns.filter(column => column === "early_menarche")
    : current?.slot === 1
      ? availableColumns.filter(column => column === "menarche_class")
      : availableColumns;
  const selectableFeatures = availableColumns.filter(column => column !== target);
  // Use only the features explicitly selected by the user.
  const selectedFeatures = useMemo(
    () => features.filter(feature => feature !== target && availableColumns.includes(feature)),
    [features, target, availableColumns]
  );
  const isMenarcheModel = current?.slot === 2 && target === "early_menarche" &&
    selectedFeatures.includes("num__EW6") &&
    selectedFeatures.includes("num__ED6");

  async function refresh() {
    try {
      const data = await api("/datasets");
      setSets(data);
      setError("");
    } catch (e) {
      setError("Backend is not running: " + e.message);
    }
  }

  useEffect(() => { refresh(); }, []);

  function clearModelResults() {
    setResult(null);
    setNetwork(null);
    setPrediction(null);
    setLastPredictionValues(null);
    setForward(null);
    setBoundary(null);
    setComparison(null);
    setWeightInfo(null);
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await api(`/datasets/${slot}/upload`, {
        method: "POST", body: form
      });
      await refresh();
      setSelected(data.dataset_id);
      setColumns(data.columns);
      setTarget(data.columns.includes("early_menarche") ? "early_menarche" : data.columns.includes("menarche_class") ? "menarche_class" : "");
      setFeatures([]);
      setPredictionValues({});
      setForwardValues({});
      setLastPredictionValues(null);
      clearModelResults();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function choose(id) {
    setSelected(id);
    clearModelResults();
    setError("");
    try {
      const data = await api(`/datasets/${id}/preview`);
      setColumns(data.columns);
      setTarget("");
      setFeatures([]);
      setPredictionValues({});
      setForwardValues({});
      setLastPredictionValues(null);
    } catch (e) {
      setError(e.message);
    }
  }

  function toggleFeature(column) {
    setFeatures(old => old.includes(column)
      ? old.filter(x => x !== column) : [...old, column]);
  }

  function readLayerConfig() {
    const parsed = layers.split(",").map(v => Number(v.trim()));
    if (!parsed.length || parsed.some(n => !Number.isInteger(n) || n < 1 || n > 512)) {
      throw new Error("Enter valid hidden layers, e.g. 8,4");
    }
    return parsed;
  }

  async function train() {
    if (!selected || !target || !selectedFeatures.length) {
      setError("Select a dataset, target and at least one input feature.");
      return;
    }

    let hiddenLayers;
    try { hiddenLayers = readLayerConfig(); }
    catch (e) { setError(e.message); return; }

    setBusy(true);
    setError("");
    clearModelResults();

    try {
      const data = await api("/train-pytorch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selected, target,
          features: selectedFeatures,
          test_size: testSize,
          hidden_layers: hiddenLayers,
          activation, learning_rate: lr,
          epochs, weight_decay: alpha,
          random_state: 42
        })
      });
      setResult(data);
      const net = await api(`/models/${data.model_id}/network`);
      setNetwork(net);
      setForwardValues(Object.fromEntries(
        selectedFeatures.map(f => [f, ""])
      ));
      setPredictionValues(Object.fromEntries(
        selectedFeatures.map(f => [f, ""])
      ));
      setMenarcheInputs({
        currentAge: "", educationYears: "", attendedSchool: "",
        religion: "", casteGroup: "", state: ""
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function makePrediction() {
    if (!result) return;
    let values = {};
    try {
      if (isMenarcheModel) {
        values = buildMenarcheValues(menarcheInputs, selectedFeatures);
      } else {
        for (const f of selectedFeatures) {
          const raw = predictionValues[f];
          if (raw === "" || raw === undefined) {
            throw new Error(`Enter a value for ${displayFeatureName(f)}.`);
          }
          if (f === "current_age_years" && (!Number.isInteger(Number(raw)) || Number(raw) < 5 || Number(raw) > 10)) throw new Error("Enter child's current age from 5 to 10 years.");
          if (f === "num__BH5B" && (!/^\d{4}$/.test(String(raw)) || Number(raw) < 1900 || Number(raw) > 2026)) {
            throw new Error("Enter the child's birth year as a four-digit year (1900–2026), e.g. 2018.");
          }
          if (f === "num__BH5A" && (!Number.isInteger(Number(raw)) || Number(raw) < 1 || Number(raw) > 12)) {
            throw new Error("Enter the child's birth month from 1 to 12.");
          }
          if (f === "num__BH6B" && (!Number.isInteger(Number(raw)) || Number(raw) < 0 || Number(raw) > 11)) {
            throw new Error("Enter additional age in months from 0 to 11.");
          }
          if (f === "num__BH5B" && Number(raw) > 2026) throw new Error("Enter a valid four-digit birth year.");
          if (f === "num__BH6A" && (Number(raw) < 5 || Number(raw) > 10)) throw new Error("Enter child's completed age from 5 to 10 years.");
          values[f] = prepareUserFeatureValue(f, raw);
        }
      }
    } catch (e) {
      setError(e.message);
      return;
    }
    try {
      setError("");
      const data = await api("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: result.model_id, values })
      });
      setPrediction(data);
      setLastPredictionValues({ ...values });
      setForwardValues(Object.fromEntries(
        Object.entries(values).map(([key, value]) => [key, String(value)])
      ));
    } catch (e) { setError(e.message); }
  }

  async function runForward() {
    if (!result) {
      setError("Train a PyTorch model before running forward propagation.");
      return;
    }

    let values = {};
    try {
      // When Predict has already run, reuse exactly the same model input values.
      // These values are already in the prepared-data scale; never validate or scale them again.
      if (lastPredictionValues && selectedFeatures.every(f =>
        Object.prototype.hasOwnProperty.call(lastPredictionValues, f))) {
        values = { ...lastPredictionValues };
      } else if (isMenarcheModel) {
        values = buildMenarcheValues(menarcheInputs, selectedFeatures);
      } else {
        for (const feature of selectedFeatures) {
          const raw = forwardValues[feature];
          if (raw === "" || raw === undefined || raw === null) {
            throw new Error(`Enter a value for ${displayFeatureName(feature)}.`);
          }
          // The prepared CSV stores BH6A/BH6B/BH5A/BH5B as standardized values.
          // Forward propagation must receive the same scale used during training.
          const numeric = Number(raw);
          if (!Number.isFinite(numeric)) {
            throw new Error(`Enter a valid numeric value for ${displayFeatureName(feature)}.`);
          }
          values[feature] = numeric;
        }
      }
    } catch (e) {
      setError(e.message);
      setForward(null);
      return;
    }

    try {
      setError("");
      setForward(null);
      const data = await api("/models/forward-pass", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: result.model_id, values })
      });
      setForward(data);
    } catch (e) {
      setError(e.message);
      setForward(null);
    }
  }

  async function makeBoundary() {
    if (!result || !featureX || !featureY) {
      setError("Choose two features for the decision boundary.");
      return;
    }
    try {
      setError("");
      setBoundary(await api(
        `/models/${result.model_id}/decision-boundary`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            feature_x: featureX, feature_y: featureY, resolution: 45
          })
        }
      ));
    } catch (e) { setBoundary(null); setError(e.message); }
  }

  async function compareModels() {
    if (!selected || !target || !selectedFeatures.length) {
      setError("Select a dataset, target and features first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const data = await api("/train-comparison", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selected,
          target,
          features: selectedFeatures,
          configurations: [[8], [16, 8], [32, 16]],
          test_size: testSize,
          activation,
          learning_rate: lr,
          epochs,
          weight_decay: alpha,
          random_state: 42
        })
      });
      setComparison(data);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  const lossData = useMemo(
    () => (result?.loss_curve || []).map((loss, i) => ({
      epoch: i + 1, loss,
      testLoss: result?.test_loss_curve?.[i]
    })),
    [result]
  );

  const boundaryPoints = useMemo(() => {
    if (!boundary) return [];
    const points = [];
    boundary.predicted_class.forEach((row, yi) => {
      row.forEach((cls, xi) => {
        points.push({
          x: boundary.x[xi],
          y: boundary.y[yi],
          class: cls
        });
      });
    });
    return points;
  }, [boundary]);

  const numericFeatures = selectedFeatures;

  return (
    <div className="app">
      <header>
        <div className="brand">
          <BrainCircuit />
          <div>
            <b>Growth & Menarche AI Playground</b>
            <small>Interactive · Explainable · Reproducible</small>
          </div>
        </div>
        <span className="status"><i /> Research workspace</span>
      </header>

      <main>
        <section className="intro">
          <div>
            <p className="eyebrow">NEURAL NETWORK LABORATORY</p>
            <h1>Explore how data shapes predictions.</h1>
            <p>
              Upload and analyse three independent datasets.
              Inspect neural network calculations and model behaviour.
            </p>
          </div>
          <BrainCircuit size={48} />
        </section>

        <div className="layout">
          <aside className="panel">
            <h2><Database size={17}/> DATASETS</h2>
            <p className="muted">
              Three prepared datasets are loaded automatically. Choose one below; no upload is needed.
            </p>

            {[1, 2, 3].map(n => (
              <div key={n}
                className={`slot ${slot === n ? "active" : ""}`}
                onClick={() => setSlot(n)}>
                <div>
                  <b>Dataset {n}</b>
                  <small>{sets.find(s => s.slot === n)?.name || "No file uploaded"}</small>
                </div>
                <span>{sets.some(s => s.slot === n) ? "Ready" : "Empty"}</span>
              </div>
            ))}

            <h3>Available datasets</h3>
            {sets.map(item => (
              <button key={item.dataset_id}
                className={`dataset ${selected === item.dataset_id ? "chosen" : ""}`}
                onClick={() => choose(item.dataset_id)}>
                Dataset {item.slot}: {item.display_name || item.name}
                <small>{item.rows?.toLocaleString()} rows · {item.columns?.length} columns</small>
              </button>
            ))}

            {current && <>
              <h3>Choose input features</h3>
              <p className="muted tiny">
                Select the columns the model should use to learn. Select All Features selects every valid input column and excludes the target and any age/metadata leakage columns. Feature names include survey meanings and code labels where available.
              </p>
              <label className="selectAllRow">
                <input
                  type="checkbox"
                  checked={selectableFeatures.length > 0 &&
                    selectableFeatures.every(c => features.includes(c))}
                  onChange={e => setFeatures(e.target.checked
                    ? selectableFeatures
                    : [])}
                />
                <b>Select all features ({selectableFeatures.length})</b>
              </label>
              <div className="columnList">
                {availableColumns.map(column => (
                  <label key={column}>
                    <input type="checkbox"
                      checked={features.includes(column)}
                      disabled={column === target}
                      onChange={() => toggleFeature(column)}/>
                    <span>{displayFeatureName(column)}</span>
                    {column === target && <small>Target column — choose above</small>}
                  </label>
                ))}
              </div>
            </>}
          </aside>

          <section className="centre">
            <Panel title="MODEL CONFIGURATION" icon={SlidersHorizontal}>
              <div className="formGrid">
                <label>
                  Target label (what the model learns to predict)
                  <select value={target} onChange={e => {
                    setTarget(e.target.value);
                    setFeatures(old => old.filter(x => x !== e.target.value));
                    clearModelResults();
                  }}>
                    <option value="">Step 1 — choose the target label</option>
                    {targetOptions.map(c => <option key={c} value={c}>{displayFeatureName(c)}</option>)}
                  </select>
                  <small>
                    {current?.slot === 1
                      ? "Dataset 1 is the small test_menarche.csv file (200 test records). Select menarche_class as the target and feature_x / feature_z as inputs to test the training, prediction and forward-propagation workflow. This is a test dataset, not IHDS research data."
                      : current?.slot === 2
                        ? "Dataset 2 (IHDS-II DS3 data.csv) assignment version: select early_menarche. Label 1 = Early menarche (synthetic menarche age below 12); label 0 = Not early menarche (12 or older). Current age is 5–10 years. The added outcome is synthetic, not an observed IHDS outcome. Synthetic age and data_origin are hidden from model inputs. Use Select all features on the left."
                        : current?.slot === 3
                          ? "Dataset 3 (IHDS-II DS4 menarche data.csv) assignment version: select early_menarche. It contains girl records with current age 5–10 years and a synthetic menarche outcome. Label 1 = Early menarche (below 12); label 0 = Not early menarche (12 or older). The added outcome is synthetic, not an observed IHDS outcome. Synthetic age and data_origin are hidden from model inputs."
                          : "Choose the outcome column you want the model to predict. Do not select the target as an input feature."}
                  </small>
                </label>
                <label>
                  Hidden layers
                  <input value={layers} onChange={e => setLayers(e.target.value)}
                    placeholder="8,4"/>
                  <small>Example: 8,4 means two hidden layers.</small>
                </label>
                <label>
                  Activation function
                  <select value={activation} onChange={e => setActivation(e.target.value)}>
                    {["relu", "tanh", "sigmoid", "leaky_relu", "elu"].map(a =>
                      <option key={a} value={a}>{a}</option>)}
                  </select>
                </label>
                <label>
                  Learning rate: {lr}
                  <input type="number" min="0.00001" max="1" step="0.0001"
                    value={lr} onChange={e => setLr(Number(e.target.value))}/>
                </label>
                <label>
                  Epochs: {epochs}
                  <input type="number" min="1" max="2000"
                    value={epochs} onChange={e => setEpochs(Number(e.target.value))}/>
                </label>
                <label>
                  Weight decay: {alpha}
                  <input type="number" min="0" step="0.0001"
                    value={alpha} onChange={e => setAlpha(Number(e.target.value))}/>
                </label>
                <label>
                  Test split: {Math.round(testSize * 100)}%
                  <input type="range" min="0.1" max="0.4" step="0.05"
                    value={testSize} onChange={e => setTestSize(Number(e.target.value))}/>
                </label>
              </div>

              <button className="train" disabled={busy || !selected}
                onClick={train}>
                <Play size={16}/> {busy ? "Processing..." : "Train PyTorch model"}
              </button>
              <button className="secondary" disabled={busy || !selected}
                onClick={compareModels}>
                <BarChart3 size={16}/> Compare hidden-layer configurations
              </button>
            </Panel>

            <Panel title="TRAINED NEURAL NETWORK" icon={Network}>
              <NetworkVisualization
                network={network}
                featureNames={result?.feature_names || selectedFeatures}
                classLabels={(result?.class_labels || []).map(value => formatClassLabel(target, value))}
                activations={forward}
                onWeightClick={setWeightInfo}/>
              {weightInfo && (
                <p className="muted">
                  Selected weight: {weightInfo.weight.toFixed(8)}
                </p>
              )}
            </Panel>

            {result && <>
              <Panel title="MODEL EVALUATION" icon={Activity}>
                <div className="metrics">
                  <Metric label="Accuracy" value={`${(result.accuracy * 100).toFixed(1)}%`}/>
                  <Metric label="Precision (macro)" value={`${(result.precision_macro * 100).toFixed(1)}%`}/>
                  <Metric label="Recall (macro)" value={`${(result.recall_macro * 100).toFixed(1)}%`}/>
                  <Metric label="F1 (macro)" value={`${(result.f1_macro * 100).toFixed(1)}%`}/>
                  <Metric label="ROC-AUC" value={result.roc_auc == null ? "N/A" : result.roc_auc.toFixed(3)}/>
                  <Metric label="Training rows" value={result.train_rows}/>
                  <Metric label="Test rows" value={result.test_rows}/>
                  <Metric label="Dataset slot" value={result.dataset_slot}/>
                </div>
                <p className="warning">{result.warning}</p>
                <h3>Training and test loss</h3>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossData}>
                      <CartesianGrid strokeDasharray="3 3"/>
                      <XAxis dataKey="epoch"/>
                      <YAxis/>
                      <Tooltip/>
                      <Line type="monotone" dataKey="loss" name="Training loss"
                        stroke="#315bd6" dot={false}/>
                      <Line type="monotone" dataKey="testLoss" name="Test loss"
                        stroke="#e58a25" dot={false}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <h3>Confusion matrix</h3>
                <div className="matrix">
                  {result.confusion_matrix?.map((row, i) => (
                    <div key={i}>
                      {row.map((value, j) => (
                        <span key={j}>{value}</span>
                      ))}
                    </div>
                  ))}
                </div>
                <p className="muted tiny">
                  Class order: {(result.class_labels || []).join(", ")}
                </p>
              </Panel>

              <Panel title="PREDICTION" icon={Activity}>
                {isMenarcheModel ? (
                  <>
                    <p className="muted">
                      Enter the respondent's original survey values. The playground
                      converts them into the 52 encoded model inputs automatically.
                    </p>
                    <MenarcheFields value={menarcheInputs}
                      onChange={setMenarcheInputs} featureNames={selectedFeatures} error={error}/>
                  </>
                ) : (
                  <FeatureInputFields featureNames={selectedFeatures}
                    values={predictionValues} onChange={setPredictionValues} error={error}/>
                )}
                {target === "early_menarche" && <p className="muted tiny">
                  Prediction labels: <b>1 = Early menarche</b> · <b>0 = Not early menarche</b>. Early menarche means age at first menstruation below 12; not early means 12 or older. The respondent current age in these assignment datasets is 5–10 years. The target outcome is synthetic.
                </p>}
                {target === "attended_school" && <p className="muted tiny">
                  Target labels: <b>1 = Attended school</b> · <b>0 = Never attended school</b>.
                </p>}
                {target === "child_is_girl" && <p className="muted tiny">
                  Target labels: <b>1 = Girl</b> · <b>0 = Boy</b>.
                </p>}
                <button className="secondary" onClick={makePrediction}>
                  Predict
                </button>
                {error && !error.toLowerCase().includes("completed age") && !error.toLowerCase().includes("current age") && <div className="error" role="alert">{error}</div>}
                {prediction && (
                  <div className="prediction">
                    <b>Predicted outcome: {prediction.predicted_label || formatClassLabel(target, prediction.predicted_class)}</b>
                    {isMenarcheModel ? (
                      <div className="metrics" style={{ marginTop: 10 }}>
                        <Metric label="Class 0 — Not early menarche"
                          value={`${(Number(prediction.probabilities?.["0"] ?? 0) * 100).toFixed(2)}%`}/>
                        <Metric label="Class 1 — Early menarche"
                          value={`${(Number(prediction.probabilities?.["1"] ?? 0) * 100).toFixed(2)}%`}/>
                      </div>
                    ) : (
                      <div className="metrics" style={{ marginTop: 10 }}>
                        {Object.entries(prediction.probabilities || {}).map(([classValue, probability]) => (
                          <Metric key={classValue}
                            label={formatClassLabel(target, classValue)}
                            value={`${(Number(probability) * 100).toFixed(2)}%`}/>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </Panel>

              <Panel title="LIVE FORWARD PROPAGATION" icon={GitBranch}>
                <p className="muted">
                  Enter one observation. The backend calculates the real
                  intermediate neuron values using the trained model.
                </p>
                {isMenarcheModel ? (
                  <MenarcheFields value={menarcheInputs}
                    onChange={setMenarcheInputs} featureNames={selectedFeatures}
                    freezeCategories={Boolean(prediction)} error={error}/>
                ) : (
                  <FeatureInputFields featureNames={selectedFeatures}
                    values={forwardValues} onChange={setForwardValues}
                    disabledCategories={Boolean(prediction)} mode="forward" error={error}/>
                )}
                <button className="secondary" onClick={runForward}>
                  <Play size={14}/> Run forward propagation
                </button>
                {error && !error.toLowerCase().includes("completed age") && !error.toLowerCase().includes("current age") && <div className="error" role="alert">{error}</div>}

                {forward?.trace?.map((layer, i) => (
                  <div className="metric" key={i} style={{ marginTop: 8 }}>
                    <b>{layer.layer}</b>
                    {layer.pre_activation && <small>
                      Before activation: {JSON.stringify(layer.pre_activation[0])}
                    </small>}
                    {layer.activation && <small>
                      After activation: {JSON.stringify(layer.activation[0])}
                    </small>}
                    {layer.values && <small>
                      Input values: {JSON.stringify(layer.values[0])}
                    </small>}
                    {layer.logits && <small>
                      Output logits: {JSON.stringify(layer.logits[0])}
                    </small>}
                    {layer.probabilities && <small>
                      Output probabilities: {JSON.stringify(layer.probabilities[0])}
                    </small>}
                  </div>
                ))}
              </Panel>

              <Panel title="DECISION BOUNDARY" icon={Eye}>
                <p className="muted">
                  Select two numeric features used by this model.
                  Other model features are held at their reference values.
                </p>
                <div className="formGrid">
                  <label>Feature X
                    <select value={featureX} onChange={e => setFeatureX(e.target.value)}>
                      <option value="">Select X</option>
                      {numericFeatures.map(f => <option key={f}>{f}</option>)}
                    </select>
                  </label>
                  <label>Feature Y
                    <select value={featureY} onChange={e => setFeatureY(e.target.value)}>
                      <option value="">Select Y</option>
                      {numericFeatures.map(f => <option key={f}>{f}</option>)}
                    </select>
                  </label>
                </div>
                <button className="secondary" onClick={makeBoundary}>
                  Generate decision boundary
                </button>
                {boundary && (
                  <>
                    <p className="muted tiny">
                      {boundary.note} Class labels: {boundary.class_labels?.join(", ")}
                    </p>
                    <DecisionBoundaryPlot boundary={boundary} featureX={featureX} featureY={featureY} />
                    <p className="muted tiny">
                      Shading shows model-predicted classes; circles show records from the uploaded dataset. These are descriptive overlays, not independent test-set validation.
                    </p>
                  </>
                )}
              </Panel>
            </>}
            
            {comparison && (
              <Panel title="TRAINING CONFIGURATION COMPARISON" icon={BarChart3}>
                <p className="muted">
                  All configurations use the same dataset, features and split.
                </p>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={(comparison.results || []).map((r) => ({
                        ...r,
                        architecture_label: `[${(r.hidden_layers || []).join(", ")}]`,
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="architecture_label"
                        interval={0}
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      />
                      <Tooltip
                        formatter={(value, name) => [
                          `${(Number(value) * 100).toFixed(2)}%`,
                          name,
                        ]}
                      />
                      <Bar dataKey="accuracy" name="Accuracy" fill="#315bd6" />
                      <Bar dataKey="f1_macro" name="F1 macro" fill="#16a36a" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="metrics">
                  {comparison.results.map((r, i) => (
                    <div className="metric" key={i}>
                      <small>Architecture [{r.hidden_layers.join(", ")}]</small>
                      <b>{(r.accuracy * 100).toFixed(1)}%</b>
                      <small>F1 macro: {r.f1_macro.toFixed(3)}</small>
                      <small>Test loss: {r.final_test_loss.toFixed(4)}</small>
                    </div>
                  ))}
                </div>
                <p className="warning">{comparison.warning}</p>
              </Panel>
            )}
          </section>
        </div>
      </main>
      <footer>
        Research and educational playground · Not a medical diagnostic system
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App/>);
