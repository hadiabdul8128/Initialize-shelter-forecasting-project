const form = document.querySelector("#estimate-form");
const dateInput = document.querySelector("#target-date");
const dateHelp = document.querySelector("#date-help");
const button = document.querySelector("#estimate-button");
const buttonLabel = button.querySelector(".button-label");
const errorMessage = document.querySelector("#form-error");
const status = document.querySelector("#model-status");
const resultCard = document.querySelector("#result-card");
const resultKicker = document.querySelector("#result-kicker");
const resultNumber = document.querySelector("#result-number");
const resultDate = document.querySelector("#result-date");
const resultDetails = document.querySelector("#result-details");
const resultRange = document.querySelector("#result-range");
const resultAsOf = document.querySelector("#result-as-of");

const numberFormat = new Intl.NumberFormat("en-US");
const dateFormat = new Intl.DateTimeFormat("en-US", {
  month: "long",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

function todayISO() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatDate(value) {
  return dateFormat.format(new Date(`${value}T00:00:00Z`));
}

function clamp(value, minimum, maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

async function initialize() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("The model is not ready.");
    const config = await response.json();

    dateInput.min = config.history_start;
    dateInput.max = config.maximum_forecast_date;
    dateInput.value = clamp(
      todayISO(),
      config.history_start,
      config.maximum_forecast_date,
    );
    dateHelp.textContent =
      `Recorded data: ${formatDate(config.history_start)}–` +
      `${formatDate(config.last_observed_date)}. Forecasts available through ` +
      `${formatDate(config.maximum_forecast_date)}.`;
    status.classList.add("is-ready");
    status.lastElementChild.textContent = "Model ready";
  } catch (error) {
    errorMessage.textContent =
      "The model could not be loaded. Refresh the page and try again.";
    status.lastElementChild.textContent = "Model unavailable";
    button.disabled = true;
  }
}

function setLoading(isLoading) {
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
  buttonLabel.textContent = isLoading ? "Calculating" : "Estimate population";
}

function showResult(data) {
  const observed = data.source === "observed";
  resultCard.classList.remove("is-empty");
  resultKicker.textContent = observed ? "Recorded census" : "PyTorch estimate";
  resultNumber.textContent = numberFormat.format(data.population);
  resultDate.textContent = `${formatDate(data.requested_date)} · people in shelter`;
  resultRange.textContent = observed
    ? "Recorded value"
    : `${numberFormat.format(data.lower_95_approx)}–${numberFormat.format(
        data.upper_95_approx,
      )}`;
  resultAsOf.textContent = formatDate(data.last_observed_date);
  resultDetails.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.textContent = "";

  if (!dateInput.value) {
    errorMessage.textContent = "Choose a date before estimating.";
    dateInput.focus();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(
      `/api/estimate?target_date=${encodeURIComponent(dateInput.value)}`,
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The estimate could not be calculated.");
    }
    showResult(data);
  } catch (error) {
    errorMessage.textContent =
      error instanceof Error
        ? error.message
        : "The estimate could not be calculated. Try again.";
  } finally {
    setLoading(false);
  }
});

initialize();
