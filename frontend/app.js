const form = document.querySelector("#plannerForm");
const submitButton = document.querySelector("#submitButton");
const statusBox = document.querySelector("#status");
const resultBox = document.querySelector("#result");
const resultTitle = document.querySelector("#resultTitle");
const copyButton = document.querySelector("#copyButton");
const agents = Array.from(document.querySelectorAll(".agent"));

let latestReport = "";
let progressTimer = null;

const progressSteps = [
  ["research", "Research agent is gathering destination context."],
  ["itinerary", "Itinerary agent is shaping the day-by-day rhythm."],
  ["budget", "Budget agent is estimating the practical numbers."],
  ["report", "Report agent is polishing everything into one plan."]
];

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = {
    destination: formData.get("destination"),
    duration_days: Number(formData.get("duration_days")),
    travel_style: formData.get("travel_style"),
    interests: formData.get("interests")
  };

  setLoading(true);
  startProgress();

  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The planner could not generate a trip.");
    }

    latestReport = data.final_report || "";
    resultTitle.textContent = `${payload.duration_days} days in ${payload.destination}`;
    resultBox.classList.remove("empty");
    resultBox.innerHTML = renderMarkdown(latestReport);
    statusBox.textContent = "Your travel plan is ready.";
    copyButton.disabled = !latestReport;
    markAllDone();
  } catch (error) {
    latestReport = "";
    resultTitle.textContent = "Something needs attention";
    resultBox.classList.remove("empty");
    resultBox.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    statusBox.textContent = "Generation stopped before the final report.";
    copyButton.disabled = true;
  } finally {
    setLoading(false);
    stopProgress();
  }
});

copyButton.addEventListener("click", async () => {
  if (!latestReport) return;
  await navigator.clipboard.writeText(latestReport);
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy";
  }, 1400);
});

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.querySelector("span:first-child").textContent = isLoading
    ? "Planning..."
    : "Generate Travel Plan";
}

function startProgress() {
  let index = 0;
  updateProgress(index);
  progressTimer = window.setInterval(() => {
    index = Math.min(index + 1, progressSteps.length - 1);
    updateProgress(index);
  }, 5200);
}

function stopProgress() {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
}

function updateProgress(activeIndex) {
  const [, message] = progressSteps[activeIndex];
  statusBox.textContent = message;

  agents.forEach((agent, index) => {
    agent.classList.toggle("is-active", index === activeIndex);
    agent.classList.toggle("is-done", index < activeIndex);
  });
}

function markAllDone() {
  agents.forEach((agent) => {
    agent.classList.remove("is-active");
    agent.classList.add("is-done");
  });
}

function renderMarkdown(markdown) {
  if (!markdown.trim()) {
    return "<p>No report was returned.</p>";
  }

  const lines = markdown.split(/\r?\n/);
  let html = "";
  let listType = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line) {
      html += closeList();
      continue;
    }

    const numbered = line.match(/^\d+\.\s+(.*)$/);
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const heading = line.match(/^#{1,3}\s+(.*)$/) || line.match(/^\*\*(.+)\*\*$/);

    if (heading) {
      html += closeList();
      html += `<h3>${formatInline(heading[1])}</h3>`;
    } else if (numbered) {
      if (listType !== "ol") html += switchList("ol");
      html += `<li>${formatInline(numbered[1])}</li>`;
    } else if (bullet) {
      if (listType !== "ul") html += switchList("ul");
      html += `<li>${formatInline(bullet[1])}</li>`;
    } else {
      html += closeList();
      html += `<p>${formatInline(line)}</p>`;
    }
  }

  html += closeList();
  return html;

  function switchList(type) {
    const closed = closeList();
    listType = type;
    return `${closed}<${type}>`;
  }

  function closeList() {
    if (!listType) return "";
    const closing = `</${listType}>`;
    listType = null;
    return closing;
  }
}

function formatInline(value) {
  return escapeHtml(value).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
