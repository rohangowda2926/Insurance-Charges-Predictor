// Simple client-side prediction using hardcoded coefficients
// This approximates the trained model for demo purposes

const MODEL_COEFFICIENTS = {
  age: 256.8,
  bmi: 339.2,
  children: 475.5,
  sex_male: -131.3,
  smoker_yes: 23848.5,
  region_northwest: -353.0,
  region_southeast: -1035.7,
  region_southwest: -960.0,
  intercept: -11938.5
};

function predict(features) {
  let prediction = MODEL_COEFFICIENTS.intercept;
  
  prediction += features.age * MODEL_COEFFICIENTS.age;
  prediction += features.bmi * MODEL_COEFFICIENTS.bmi;
  prediction += features.children * MODEL_COEFFICIENTS.children;
  
  if (features.sex === 'male') {
    prediction += MODEL_COEFFICIENTS.sex_male;
  }
  
  if (features.smoker === 'yes') {
    prediction += MODEL_COEFFICIENTS.smoker_yes;
  }
  
  if (features.region === 'northwest') {
    prediction += MODEL_COEFFICIENTS.region_northwest;
  } else if (features.region === 'southeast') {
    prediction += MODEL_COEFFICIENTS.region_southeast;
  } else if (features.region === 'southwest') {
    prediction += MODEL_COEFFICIENTS.region_southwest;
  }
  
  return Math.max(0, prediction);
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const resultDiv = document.getElementById("result");
  const insightDiv = document.getElementById("insight");
  const insightText = document.getElementById("insight-text");
  const chipRow = document.getElementById("chip-row");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const payload = {
      age: Number(formData.get("age")),
      sex: formData.get("sex"),
      bmi: Number(formData.get("bmi")),
      children: Number(formData.get("children")),
      smoker: formData.get("smoker"),
      region: formData.get("region"),
    };

    const button = form.querySelector("button");
    const originalText = button.textContent;

    button.textContent = "Predicting...";
    button.disabled = true;
    resultDiv.style.display = "none";
    insightDiv.textContent = "";
    chipRow.innerHTML = "";

    try {
      // Use client-side prediction
      const amountNumber = predict(payload);

      const amount = amountNumber.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });

      // Show main result
      resultDiv.style.display = "block";
      resultDiv.className = "result";
      resultDiv.textContent = `Predicted yearly charges: ${amount}`;

      // Derive a simple risk band from the prediction
      let band = "";
      let bandLabel = "";
      let explanation = "";

      if (amountNumber < 8000) {
        band = "low";
        bandLabel = "Low risk";
        explanation =
          "Your predicted charges are on the lower side compared to typical policy holders.";
      } else if (amountNumber < 20000) {
        band = "medium";
        bandLabel = "Moderate risk";
        explanation =
          "Your predicted charges sit in a moderate band. Lifestyle improvements may reduce future costs.";
      } else {
        band = "high";
        bandLabel = "Higher risk";
        explanation =
          "Your predicted charges are relatively high. Risk factors like smoking, high BMI or age strongly influence this.";
      }

      // Update right-hand insight panel
      const chips = [];

      chips.push(`<span class="chip ${band}">${bandLabel}</span>`);
      chips.push(
        `<span class="chip">${payload.smoker === "yes" ? "Smoker" : "Non-smoker"}</span>`
      );
      chips.push(`<span class="chip">BMI: ${payload.bmi}</span>`);
      chips.push(`<span class="chip">Age: ${payload.age}</span>`);

      chipRow.innerHTML = chips.join("");
      insightText.textContent = explanation;
      insightDiv.textContent =
        "These bands are for demonstration only and are not medical or financial advice.";
    } catch (err) {
      console.error(err);
      resultDiv.style.display = "block";
      resultDiv.className = "result error";
      resultDiv.textContent = "Something went wrong while predicting.";
      insightText.textContent =
        "An error occurred. Please try again.";
    } finally {
      button.textContent = originalText;
      button.disabled = false;
    }
  });
});