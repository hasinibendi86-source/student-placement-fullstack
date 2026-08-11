/* ============================================================
   script.js
   ------------------------------------------------------------
   Handles form validation and talks to the Flask backend
   (POST /predict) using the Fetch API, then renders the result.

   IMPORTANT: If your Flask server is running somewhere other
   than http://127.0.0.1:5000, update API_URL below.
   ============================================================ */

const API_URL = "http://127.0.0.1:5000/predict";

// ----- Grab elements -----
const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const btnLabel = predictBtn.querySelector(".btn__label");
const btnSpinner = predictBtn.querySelector(".btn__spinner");
const resetBtn = document.getElementById("reset-btn");

const formBanner = document.getElementById("form-banner");

const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const verdictStamp = document.getElementById("verdict-stamp");
const probabilityValue = document.getElementById("probability-value");
const probabilityFill = document.getElementById("probability-fill");
const resultSummary = document.getElementById("result-summary");

// ----- Field definitions used for validation -----
const FIELDS = [
  { id: "cgpa", label: "CGPA", min: 0, max: 10 },
  { id: "tenth", label: "10th Percentage", min: 0, max: 100 },
  { id: "twelfth", label: "12th Percentage", min: 0, max: 100 },
  { id: "technical", label: "Technical Skill Score", min: 0, max: 100 },
  { id: "aptitude", label: "Aptitude Score", min: 0, max: 100 },
  { id: "communication", label: "Communication Score", min: 0, max: 100 },
];

/**
 * Validates all number fields + the internship dropdown.
 * Shows inline error messages next to each invalid field.
 * Returns true if the whole form is valid.
 */
function validateForm() {
  let isValid = true;

  FIELDS.forEach(({ id, label, min, max }) => {
    const input = document.getElementById(id);
    const errorEl = document.getElementById(`err-${id}`);
    const rawValue = input.value.trim();
    const value = parseFloat(rawValue);

    input.classList.remove("invalid");
    errorEl.textContent = "";

    if (rawValue === "") {
      errorEl.textContent = `${label} is required.`;
      input.classList.add("invalid");
      isValid = false;
    } else if (Number.isNaN(value)) {
      errorEl.textContent = `${label} must be a number.`;
      input.classList.add("invalid");
      isValid = false;
    } else if (value < min || value > max) {
      errorEl.textContent = `${label} must be between ${min} and ${max}.`;
      input.classList.add("invalid");
      isValid = false;
    }
  });

  // Internship dropdown
  const internshipInput = document.getElementById("internship");
  const internshipError = document.getElementById("err-internship");
  internshipInput.classList.remove("invalid");
  internshipError.textContent = "";

  if (internshipInput.value === "") {
    internshipError.textContent = "Please select an option.";
    internshipInput.classList.add("invalid");
    isValid = false;
  }

  return isValid;
}

/** Shows/hides the loading state on the Predict button. */
function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Predicting..." : "Predict Placement";
}

/** Shows a red banner above the buttons for server/network errors. */
function showBanner(message) {
  formBanner.textContent = message;
  formBanner.hidden = false;
}

function hideBanner() {
  formBanner.hidden = true;
  formBanner.textContent = "";
}

/** Renders the prediction result into the result card. */
function renderResult(data, formValues) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  const isPlaced = data.prediction === "Placed";

  // Verdict stamp
  verdictStamp.textContent = isPlaced ? "PLACED" : "NOT PLACED";
  verdictStamp.classList.remove("is-placed", "is-not-placed", "animate-in");
  verdictStamp.classList.add(isPlaced ? "is-placed" : "is-not-placed");
  // Force reflow so the animation re-triggers on repeated predictions
  void verdictStamp.offsetWidth;
  verdictStamp.classList.add("animate-in");

  // Probability meter
  const probability = data.probability;
  probabilityValue.textContent = `${probability}%`;
  probabilityFill.style.width = `${Math.min(Math.max(probability, 0), 100)}%`;

  // Summary of what was submitted
  resultSummary.innerHTML = `
    <div><dt>CGPA</dt><dd>${formValues.cgpa}</dd></div>
    <div><dt>10th / 12th</dt><dd>${formValues.tenth}% / ${formValues.twelfth}%</dd></div>
    <div><dt>Technical Score</dt><dd>${formValues.technical}</dd></div>
    <div><dt>Aptitude Score</dt><dd>${formValues.aptitude}</dd></div>
    <div><dt>Communication</dt><dd>${formValues.communication}</dd></div>
    <div><dt>Internship</dt><dd>${formValues.internship === "1" ? "Yes" : "No"}</dd></div>
  `;
}

/** Resets the result card back to its empty state. */
function resetResult() {
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  hideBanner();
}

// ----- Form submit handler -----
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideBanner();

  if (!validateForm()) {
    return; // stop here if any field is invalid
  }

  const formValues = {
    cgpa: document.getElementById("cgpa").value,
    tenth: document.getElementById("tenth").value,
    twelfth: document.getElementById("twelfth").value,
    technical: document.getElementById("technical").value,
    aptitude: document.getElementById("aptitude").value,
    communication: document.getElementById("communication").value,
    internship: document.getElementById("internship").value,
  };

  const payload = {
    cgpa: parseFloat(formValues.cgpa),
    tenth: parseFloat(formValues.tenth),
    twelfth: parseFloat(formValues.twelfth),
    technical: parseFloat(formValues.technical),
    aptitude: parseFloat(formValues.aptitude),
    communication: parseFloat(formValues.communication),
    internship: parseInt(formValues.internship, 10),
  };

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    // Try to parse JSON even on error responses, since the backend
    // returns helpful error messages in JSON format.
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const message = (data && data.error) ? data.error : `Request failed (status ${response.status}).`;
      showBanner(message);
      resultEmpty.hidden = false;
      resultContent.hidden = true;
      return;
    }

    if (!data || typeof data.probability !== "number") {
      showBanner("Received an unexpected response from the server.");
      return;
    }

    renderResult(data, formValues);

  } catch (networkError) {
    // This fires if Flask isn't running, or the frontend can't reach it (CORS, offline, etc.)
    showBanner(
      "Could not reach the prediction server. Make sure the Flask backend is running at " +
      API_URL.replace("/predict", "") + "."
    );
  } finally {
    setLoading(false);
  }
});

// ----- Reset button handler -----
resetBtn.addEventListener("click", () => {
  // Let the native form reset run first, then clear our own state
  setTimeout(() => {
    FIELDS.forEach(({ id }) => {
      document.getElementById(id).classList.remove("invalid");
      document.getElementById(`err-${id}`).textContent = "";
    });
    document.getElementById("internship").classList.remove("invalid");
    document.getElementById("err-internship").textContent = "";
    resetResult();
  }, 0);
});
