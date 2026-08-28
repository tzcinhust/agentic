const entries = [
  {
    id: "main-01",
    track: "main",
    model: "GPT-5.4",
    reasoningLabel: "high",
    agent: "",
    organization: "OpenAI",
    submissionDate: "2026-05-25",
    benchmarkVersion: "0.7.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 55.7,
      overallPassAt1Std: 1.9,
      passAt5: 38.0,
      meanUxScore: 3.49,
      costPerTask: 0.0809,
      domains: {
        travel: { passAt1: 55.9, passAt1Std: 2.8, passAt5: 36.0, meanUxScore: 3.37, costPerTask: 0.1224 },
        customerSupport: { passAt1: 57.6, passAt1Std: 2.5, passAt5: 38.0, meanUxScore: 3.49, costPerTask: 0.0716 },
        shoppingAssistant: { passAt1: 53.6, passAt1Std: 2.1, passAt5: 40.0, meanUxScore: 3.61, costPerTask: 0.0486 },
      },
    },
  },
  {
    id: "main-02",
    track: "main",
    model: "Claude Opus 4.7",
    reasoningLabel: "high",
    agent: "",
    organization: "Anthropic",
    submissionDate: "2026-05-29",
    benchmarkVersion: "0.7.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 53.4,
      overallPassAt1Std: 1.7,
      passAt5: 33.9,
      meanUxScore: 3.46,
      costPerTask: 0.0773,
      domains: {
        travel: { passAt1: 55.0, passAt1Std: 1.0, passAt5: 36.0, meanUxScore: 3.47, costPerTask: 0.1068 },
        customerSupport: { passAt1: 51.0, passAt1Std: 3.0, passAt5: 28.0, meanUxScore: 3.19, costPerTask: 0.0755 },
        shoppingAssistant: { passAt1: 54.0, passAt1Std: 1.0, passAt5: 37.0, meanUxScore: 3.72, costPerTask: 0.0496 },
      },
    },
  },
  {
    id: "main-03",
    track: "main",
    model: "Kimi-K2.6",
    agent: "",
    organization: "Moonshot AI",
    submissionDate: "2026-05-25",
    benchmarkVersion: "0.7.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 48.3,
      overallPassAt1Std: 2.1,
      passAt5: 29.3,
      meanUxScore: 3.37,
      costPerTask: 0.0496,
      domains: {
        travel: { passAt1: 51.9, passAt1Std: 4.0, passAt5: 26.0, meanUxScore: 3.22, costPerTask: 0.0871 },
        customerSupport: { passAt1: 45.1, passAt1Std: 1.8, passAt5: 26.0, meanUxScore: 3.35, costPerTask: 0.0339 },
        shoppingAssistant: { passAt1: 47.9, passAt1Std: 2.2, passAt5: 36.0, meanUxScore: 3.54, costPerTask: 0.0279 },
      },
    },
  },
  {
    id: "main-04",
    track: "main",
    model: "DeepSeek-v4-Pro",
    agent: "",
    organization: "DeepSeek",
    submissionDate: "2026-05-25",
    benchmarkVersion: "0.7.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 47.2,
      overallPassAt1Std: 0.6,
      passAt5: 25.3,
      meanUxScore: 3.36,
      costPerTask: undefined,
      domains: {
        travel: { passAt1: 47.6, passAt1Std: 2.7, passAt5: 22.0, meanUxScore: 3.04, costPerTask: undefined },
        customerSupport: { passAt1: 45.6, passAt1Std: 1.4, passAt5: 23.0, meanUxScore: 3.50, costPerTask: undefined },
        shoppingAssistant: { passAt1: 48.4, passAt1Std: 2.2, passAt5: 31.0, meanUxScore: 3.54, costPerTask: undefined },
      },
    },
  },
  {
    id: "main-05",
    track: "main",
    organization: "OpenAI",
    model: "GPT-5.4",
    reasoningLabel: "default",
    agent: "",
    submissionDate: "2026-05-25",
    benchmarkVersion: "0.7.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 46.9,
      overallPassAt1Std: 0.9,
      passAt5: 26.2,
      meanUxScore: 3.41,
      costPerTask: 0.0351,
      domains: {
        travel: { passAt1: 43.2, passAt1Std: 1.5, passAt5: 22.9, meanUxScore: 3.19, costPerTask: 0.0565 },
        customerSupport: { passAt1: 47.2, passAt1Std: 2.1, passAt5: 28.1, meanUxScore: 3.49, costPerTask: 0.0271 },
        shoppingAssistant: { passAt1: 50.3, passAt1Std: 1.3, passAt5: 30.8, meanUxScore: 3.55, costPerTask: 0.0216 },
      },
    },
  },
  {
    id: "main-06",
    track: "main",
    model: "GPT-5.5",
    reasoningLabel: "high",
    agent: "",
    organization: "OpenAI",
    submissionDate: "2026-07-01",
    benchmarkVersion: "0.8.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 58.9,
      overallPassAt1Std: 0.6,
      passAt5: 44.2,
      meanUxScore: 3.62,
      costPerTask: 0.1842,
      domains: {
        travel: { passAt1: 62.0, passAt1Std: 2.0, passAt5: 45.0, meanUxScore: 3.61, costPerTask: 0.2802 },
        customerSupport: { passAt1: 59.0, passAt1Std: 1.0, passAt5: 43.0, meanUxScore: 3.38, costPerTask: 0.1553 },
        shoppingAssistant: { passAt1: 55.0, passAt1Std: 1.0, passAt5: 45.0, meanUxScore: 3.87, costPerTask: 0.1170 },
      },
    },
  },
  {
    id: "main-07",
    track: "main",
    model: "GPT-5.4",
    reasoningLabel: "high",
    agent: "",
    organization: "OpenAI",
    submissionDate: "2026-07-01",
    benchmarkVersion: "0.8.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 58.6,
      overallPassAt1Std: 0.5,
      passAt5: 41.8,
      meanUxScore: 3.67,
      costPerTask: 0.0874,
      domains: {
        travel: { passAt1: 62.0, passAt1Std: 1.0, passAt5: 39.0, meanUxScore: 3.58, costPerTask: 0.1339 },
        customerSupport: { passAt1: 59.0, passAt1Std: 2.0, passAt5: 42.0, meanUxScore: 3.59, costPerTask: 0.0808 },
        shoppingAssistant: { passAt1: 55.0, passAt1Std: 1.0, passAt5: 45.0, meanUxScore: 3.85, costPerTask: 0.0475 },
      },
    },
  },
  {
    id: "memory-01",
    track: "memory",
    model: "GPT 5.1 + Foundry Memory",
    agent: "",
    organization: "Microsoft Foundry",
    submissionDate: "2026-06-08",
    benchmarkVersion: "0.4.4",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 58.3,
      overallPassAt1Std: 4.0,
      passAt5: 37.3,
      meanUxScore: 3.93,
      costPerTask: 0.036,
      domains: {
        travel: { passAt1: 60.0, passAt1Std: 6.0, passAt5: 34.0, meanUxScore: 4.03, costPerTask: 0.065 },
        customerSupport: { passAt1: 58.0, passAt1Std: 4.0, passAt5: 36.0, meanUxScore: 4.03, costPerTask: 0.028 },
        shoppingAssistant: { passAt1: 57.0, passAt1Std: 2.0, passAt5: 42.0, meanUxScore: 3.74, costPerTask: 0.016 },
      },
    },
  },
  {
    id: "memory-02",
    track: "memory",
    model: "GPT 5.1",
    reasoningLabel: "no-memory",
    agent: "",
    organization: "OpenAI",
    submissionDate: "2026-06-08",
    benchmarkVersion: "0.4.4",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 53.3,
      overallPassAt1Std: 3.3,
      passAt5: 32.7,
      meanUxScore: 3.87,
      costPerTask: 0.031,
      domains: {
        travel: { passAt1: 56.0, passAt1Std: 5.0, passAt5: 30.0, meanUxScore: 3.96, costPerTask: 0.061 },
        customerSupport: { passAt1: 51.0, passAt1Std: 2.0, passAt5: 30.0, meanUxScore: 3.84, costPerTask: 0.017 },
        shoppingAssistant: { passAt1: 53.0, passAt1Std: 3.0, passAt5: 38.0, meanUxScore: 3.81, costPerTask: 0.014 },
      },
    },
  },
  {
    id: "memory-03",
    track: "memory",
    model: "GPT 5.4 + Foundry Memory",
    reasoningLabel: "default",
    agent: "",
    organization: "Microsoft Foundry",
    submissionDate: "2026-08-03",
    benchmarkVersion: "0.8.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 54.5,
      overallPassAt1Std: 3.0,
      passAt5: 33.6,
      meanUxScore: 3.75,
      costPerTask: undefined,
      domains: {
        travel: { passAt1: 58.6, passAt1Std: 4.0, passAt5: 30.0, meanUxScore: 3.40, costPerTask: undefined },
        customerSupport: { passAt1: 52.8, passAt1Std: 3.0, passAt5: 32.8, meanUxScore: 4.06, costPerTask: undefined },
        shoppingAssistant: { passAt1: 52.0, passAt1Std: 2.0, passAt5: 38.0, meanUxScore: 3.80, costPerTask: undefined },
      },
    },
  },
  {
    id: "memory-04",
    track: "memory",
    model: "GPT 5.4",
    reasoningLabel: "default",
    agent: "",
    organization: "OpenAI",
    submissionDate: "2026-08-03",
    benchmarkVersion: "0.8.0",
    verificationStatus: "verified",
    metrics: {
      overallPassAt1: 51.3,
      overallPassAt1Std: 3.0,
      passAt5: 29.9,
      meanUxScore: 3.31,
      costPerTask: undefined,
      domains: {
        travel: { passAt1: 55.2, passAt1Std: 4.0, passAt5: 26.4, meanUxScore: 3.30, costPerTask: undefined },
        customerSupport: { passAt1: 47.6, passAt1Std: 3.0, passAt5: 25.2, meanUxScore: 3.20, costPerTask: undefined },
        shoppingAssistant: { passAt1: 51.0, passAt1Std: 2.0, passAt5: 38.0, meanUxScore: 3.43, costPerTask: undefined },
      },
    },
  },
];

const state = {
  track: "main",
  scoreView: "overall",
  sortKey: "selectedScore",
  sortDirection: "desc",
  showAll: false,
};

const body = document.querySelector("#leaderboard-body");
const toggleRows = document.querySelector("#toggle-rows");
const tabs = Array.from(document.querySelectorAll(".track-tab"));
const scoreViewInputs = Array.from(document.querySelectorAll('input[name="score-view"]'));
const sortButtons = Array.from(document.querySelectorAll(".sort-button"));
const primaryScoreSort = document.querySelector("#primary-score-sort");

const trackUrlSlugs = {
  main: "main",
  memory: "agent-learning",
};

const trackUrlAliases = {
  main: "main",
  memory: "memory",
  "agent-learning": "memory",
};

const scoreLabels = {
  overall: "pass@1 (%)",
  travel: "pass@1 (%)",
  customerSupport: "pass@1 (%)",
  shoppingAssistant: "pass@1 (%)",
};

function selectedScore(entry) {
  if (state.scoreView === "overall") return entry.metrics.overallPassAt1;
  return entry.metrics.domains[state.scoreView].passAt1;
}

function selectedMetrics(entry) {
  if (state.scoreView === "overall") return entry.metrics;
  return entry.metrics.domains[state.scoreView];
}

function metricValue(entry, key) {
  if (key === "selectedScore") return selectedScore(entry);
  if (key === "submissionDate") return new Date(entry.submissionDate).getTime();
  if (["passAt5", "meanUxScore", "costPerTask"].includes(key)) return selectedMetrics(entry)[key];
  return entry.metrics[key];
}

function compareMetric(a, b) {
  const aValue = metricValue(a, state.sortKey);
  const bValue = metricValue(b, state.sortKey);

  if (aValue == null && bValue == null) return 0;
  if (aValue == null) return 1;
  if (bValue == null) return -1;

  const direction = state.sortDirection === "asc" ? 1 : -1;
  return aValue > bValue ? direction : aValue < bValue ? -direction : 0;
}

function compareVersions(a, b) {
  return b.localeCompare(a, undefined, { numeric: true, sensitivity: "base" });
}

function trackEntries() {
  return entries.filter((entry) => entry.track === state.track);
}

function groupedEntries(trackEntries) {
  const unversioned = trackEntries.some((entry) => !entry.benchmarkVersion);
  if (unversioned) return [{ version: null, entries: trackEntries.sort(compareMetric) }];

  const groups = new Map();
  trackEntries.forEach((entry) => {
    const group = groups.get(entry.benchmarkVersion) || [];
    group.push(entry);
    groups.set(entry.benchmarkVersion, group);
  });

  return Array.from(groups.entries())
    .sort(([a], [b]) => compareVersions(a, b))
    .map(([version, group]) => ({ version, entries: group.sort(compareMetric) }));
}

function visibleGroups(groups) {
  if (state.showAll) return groups;
  return groups.map((group) => ({ ...group, entries: group.entries.slice(0, 10) }));
}

function totalEntryCount(groups) {
  return groups.reduce((total, group) => total + group.entries.length, 0);
}

function renderEntryRow(entry, rank) {
  const metrics = selectedMetrics(entry);
  return `
    <tr>
      <td class="rank-cell"><span class="rank-pill ${rank <= 3 ? "top" : ""}">${rank}</span></td>
      <td>
        <span class="model-name">${entry.model}${entry.reasoningLabel ? ` <span class="model-variant">(${entry.reasoningLabel})</span>` : ""}</span>
      </td>
      <td>${entry.organization}</td>
      <td class="metric">${formatPassAt1(entry)}</td>
      <td class="metric">${formatPercent(metrics.passAt5)}</td>
      <td class="metric">${formatUx(metrics.meanUxScore)}</td>
      <td class="metric">${formatCost(metrics.costPerTask)}</td>
      <td>${entry.submissionDate}</td>
      <td><span class="status-badge status-${entry.verificationStatus}">${statusLabel(entry.verificationStatus)}</span></td>
    </tr>
  `;
}

function renderGroupedRows(groups) {
  return groups
    .map((group) => {
      const rows = group.entries.map((entry, index) => renderEntryRow(entry, index + 1)).join("");
      if (!group.version) return rows;

      return `
        <tr class="version-row">
          <td colspan="9">Benchmark Version: ${group.version}</td>
        </tr>
        ${rows}
      `;
    })
    .join("");
}

function formatPercent(value) {
  return value.toFixed(1);
}

function formatPassAt1(entry) {
  const metrics = selectedMetrics(entry);
  const mean = formatPercent(selectedScore(entry));
  const std = state.scoreView === "overall" ? entry.metrics.overallPassAt1Std : metrics.passAt1Std;

  if (std == null) return `<span class="metric-main">${mean}</span>`;
  return `<span class="metric-main">${mean}</span><span class="metric-std">&plusmn; ${formatPercent(std)}</span>`;
}

function formatUx(value) {
  return value.toFixed(2);
}

function formatCost(value) {
  return value == null ? "-" : `$${value.toFixed(2)}`;
}

function statusLabel(status) {
  return status.replace("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function trackFromUrl() {
  const track = new URLSearchParams(window.location.search).get("track");
  return trackUrlAliases[track] || "main";
}

function setTrackUrl(track) {
  const url = new URL(window.location.href);
  url.searchParams.set("track", trackUrlSlugs[track]);
  window.history.replaceState({}, "", url);
}

state.track = trackFromUrl();

function render() {
  const groups = groupedEntries(trackEntries());
  const visible = visibleGroups(groups);
  const totalEntries = totalEntryCount(groups);

  primaryScoreSort.textContent = scoreLabels[state.scoreView];

  body.innerHTML = renderGroupedRows(visible);

  if (!totalEntries) {
    body.innerHTML = '<tr><td class="empty-state" colspan="9">No submitted results for this track yet.</td></tr>';
  }

  toggleRows.hidden = groups.every((group) => group.entries.length <= 10);
  toggleRows.textContent = state.showAll ? "Show top 10 per version" : `Show all ${totalEntries}`;

  tabs.forEach((tab) => {
    const active = tab.dataset.track === state.track;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });

  scoreViewInputs.forEach((input) => {
    input.checked = input.value === state.scoreView;
  });

  sortButtons.forEach((button) => {
    const active = button.dataset.sort === state.sortKey;
    button.classList.toggle("active", active);
    button.classList.toggle("asc", active && state.sortDirection === "asc");
  });

}

tabs.forEach((tab) => {
  tab.addEventListener("click", (event) => {
    event.preventDefault();
    state.track = tab.dataset.track;
    state.showAll = false;
    setTrackUrl(state.track);
    render();
  });
});

scoreViewInputs.forEach((input) => {
  input.addEventListener("change", () => {
    state.scoreView = input.value;
    state.sortKey = "selectedScore";
    state.sortDirection = "desc";
    state.showAll = false;
    render();
  });
});

sortButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextKey = button.dataset.sort;
    const defaultDirection = nextKey === "costPerTask" ? "asc" : "desc";

    if (state.sortKey === nextKey) {
      state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = nextKey;
      state.sortDirection = defaultDirection;
    }

    render();
  });
});

toggleRows.addEventListener("click", () => {
  state.showAll = !state.showAll;
  render();
});

render();
