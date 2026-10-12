const form = document.querySelector("#complaint-form");
const textarea = document.querySelector("#complaint");
const count = document.querySelector("#char-count");
const resultCard = document.querySelector("#result");
const submitButton = form.querySelector("button[type='submit']");
const marathiConfirm = document.querySelector("#marathi-confirm");
const confirmMarathi = document.querySelector("#confirm-marathi");

textarea.addEventListener("input", () => {
  count.textContent = `${textarea.value.length.toLocaleString()} / 2,000`;
  const hasDevanagari = /[\u0900-\u097f]/u.test(textarea.value);
  marathiConfirm.hidden = !hasDevanagari;
  if (!hasDevanagari) confirmMarathi.checked = false;
});

document.querySelectorAll(".example-chip").forEach((button) => {
  button.addEventListener("click", () => {
    textarea.value = button.dataset.example;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
  });
});

function showResult({ status, category, department, message, error, review_reason, detected_script }) {
  resultCard.replaceChildren();
  resultCard.className = "result-card";
  const label = document.createElement("div");
  label.className = "result-label";
  const title = document.createElement("div");
  title.className = "result-title";
  const detail = document.createElement("p");
  detail.className = "result-detail";

  if (error) {
    resultCard.classList.add("error");
    label.textContent = "Could not classify";
    title.textContent = "Please try again";
    detail.textContent = error;
  } else if (status === "needs_review") {
    resultCard.classList.add("review");
    if (review_reason === "unsupported language") {
      label.textContent = "Devanagari text detected";
      title.textContent = "Manual review required";
    } else if (review_reason === "language confirmation required") {
      label.textContent = "Devanagari text detected";
      title.textContent = "Confirm Marathi to classify";
    } else {
      label.textContent = "Clarification / manual review";
      title.textContent = "A little more context would help";
    }
    detail.textContent = message || "This complaint needs a person to check it before routing.";
  } else {
    label.textContent = "Suggested category";
    title.textContent = category;
    detail.textContent = `Suggested destination: ${department}. Please review before taking action.`;
  }
  if (detected_script) detail.textContent += ` Detected script: ${detected_script}.`;
  resultCard.append(label, title, detail);
  resultCard.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultCard.hidden = true;
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Checking complaint…";
  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        complaint: textarea.value,
        language: confirmMarathi.checked ? "mr" : null,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request could not be completed.");
    showResult(data);
  } catch (error) {
    showResult({ error: error.message || "Unable to connect to the classifier." });
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Classify complaint";
  }
});
