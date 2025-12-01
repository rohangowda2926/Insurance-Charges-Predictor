document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const resultsContainer = document.getElementById("results");
  const predictedAmount = document.getElementById("predicted-amount");
  const riskLevel = document.getElementById("risk-level");
  const insights = document.getElementById("insights");

  // Enhanced form submission with validation
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    const formData = new FormData(form);
    const payload = {
      age: Number(formData.get("age")),
      sex: formData.get("sex"),
      bmi: Number(formData.get("bmi")),
      children: Number(formData.get("children")),
      smoker: formData.get("smoker"),
      region: formData.get("region"),
    };

    const button = form.querySelector(".btn-predict");
    const buttonText = button.querySelector("span:last-child");
    const buttonIcon = button.querySelector("i");
    const originalText = buttonText.textContent;
    const originalIcon = buttonIcon.className;

    // Loading state with animation
    buttonText.textContent = "Analyzing...";
    buttonIcon.className = "loading";
    button.disabled = true;
    resultsContainer.classList.remove("show");

    // Add subtle shake animation to form
    form.style.animation = "none";
    setTimeout(() => {
      form.style.animation = "pulse 0.5s ease";
    }, 10);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Request failed");

      const data = await res.json();
      const amountNumber = data.predicted_charge;

      // Animate number counting up
      animateNumber(predictedAmount, 0, amountNumber, 1500);

      // Determine risk level and styling
      let riskClass, riskText, riskIcon;
      if (amountNumber < 8000) {
        riskClass = "risk-low";
        riskText = "Low Risk";
        riskIcon = "fas fa-shield-alt";
      } else if (amountNumber < 20000) {
        riskClass = "risk-medium";
        riskText = "Moderate Risk";
        riskIcon = "fas fa-exclamation-triangle";
      } else {
        riskClass = "risk-high";
        riskText = "High Risk";
        riskIcon = "fas fa-exclamation-circle";
      }

      // Update risk indicator
      riskLevel.className = `risk-indicator ${riskClass}`;
      riskLevel.innerHTML = `<i class="${riskIcon}"></i><span>${riskText}</span>`;

      // Generate detailed insights
      generateInsights(payload, amountNumber);

      // Show results with animation
      setTimeout(() => {
        resultsContainer.classList.add("show");
      }, 500);

      // Success feedback
      showNotification("Prediction completed successfully!", "success");

    } catch (err) {
      console.error(err);
      showNotification("Failed to get prediction. Please try again.", "error");
    } finally {
      // Reset button state
      setTimeout(() => {
        buttonText.textContent = originalText;
        buttonIcon.className = originalIcon;
        button.disabled = false;
      }, 1000);
    }
  });

  // Animate number counting
  function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;

    function updateNumber(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeProgress = easeOutCubic(progress);
      const current = start + (range * easeProgress);
      
      element.textContent = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(current);

      if (progress < 1) {
        requestAnimationFrame(updateNumber);
      }
    }

    requestAnimationFrame(updateNumber);
  }

  // Easing function
  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  // Generate detailed insights
  function generateInsights(payload, prediction) {
    const insightsData = [
      {
        label: "Age Factor",
        value: getAgeFactor(payload.age),
        icon: "fas fa-birthday-cake"
      },
      {
        label: "BMI Impact",
        value: getBMICategory(payload.bmi),
        icon: "fas fa-weight"
      },
      {
        label: "Smoking Effect",
        value: payload.smoker === "yes" ? "+320%" : "Baseline",
        icon: "fas fa-smoking"
      },
      {
        label: "Family Size",
        value: payload.children === 0 ? "Individual" : `${payload.children} dependents`,
        icon: "fas fa-users"
      }
    ];

    insights.innerHTML = insightsData.map(insight => `
      <div class="insight-item">
        <i class="${insight.icon}" style="color: var(--primary); margin-bottom: 0.5rem;"></i>
        <div class="insight-value">${insight.value}</div>
        <div class="insight-label">${insight.label}</div>
      </div>
    `).join('');
  }

  // Helper functions
  function getAgeFactor(age) {
    if (age < 25) return "Young Adult";
    if (age < 40) return "Prime Age";
    if (age < 55) return "Middle Age";
    return "Senior";
  }

  function getBMICategory(bmi) {
    if (bmi < 18.5) return "Underweight";
    if (bmi < 25) return "Normal";
    if (bmi < 30) return "Overweight";
    return "Obese";
  }

  // Notification system
  function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
      <span>${message}</span>
    `;
    
    // Add notification styles
    Object.assign(notification.style, {
      position: 'fixed',
      top: '20px',
      right: '20px',
      padding: '1rem 1.5rem',
      borderRadius: '0.75rem',
      color: 'white',
      fontWeight: '500',
      zIndex: '1000',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      transform: 'translateX(100%)',
      transition: 'transform 0.3s ease',
      background: type === 'success' ? 'var(--success)' : 'var(--danger)'
    });
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
      notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
      notification.style.transform = 'translateX(100%)';
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  }

  // Enhanced form validation
  const inputs = form.querySelectorAll('input, select');
  inputs.forEach(input => {
    input.addEventListener('input', validateField);
    input.addEventListener('blur', validateField);
  });

  function validateField(e) {
    const field = e.target;
    const value = parseFloat(field.value);
    const fieldContainer = field.closest('.field');
    let existingWarning = fieldContainer.querySelector('.field-warning');
    
    // Remove existing warnings
    if (existingWarning) {
      existingWarning.remove();
    }
    field.classList.remove('warning');

    // Validation and warnings
    if (field.name === 'age' && field.value) {
      if (value < 18 || value > 64) {
        showFieldWarning(fieldContainer, '⚠ Model trained on ages 18-64. Predictions may be less reliable.');
        field.classList.add('warning');
      }
    } else if (field.name === 'bmi' && field.value) {
      if (value < 15.96 || value > 53.13) {
        showFieldWarning(fieldContainer, '⚠ Model trained on BMI 15.96-53.13. Predictions may be less reliable.');
        field.classList.add('warning');
      }
    }
  }

  function showFieldWarning(container, message) {
    const warning = document.createElement('div');
    warning.className = 'field-warning';
    warning.innerHTML = `<i class="fas fa-exclamation-triangle"></i><span>${message}</span>`;
    container.appendChild(warning);
  }

  function validateForm() {
    const formData = new FormData(form);
    const errors = [];

    // Check required fields
    if (!formData.get('age') || formData.get('age') <= 0) errors.push('Age is required and must be positive');
    if (!formData.get('sex')) errors.push('Gender must be selected');
    if (!formData.get('bmi') || formData.get('bmi') <= 0) errors.push('BMI is required and must be positive');
    if (!formData.get('children') || formData.get('children') < 0) errors.push('Children must be 0 or greater');
    if (!formData.get('smoker')) errors.push('Smoking status must be selected');
    if (!formData.get('region')) errors.push('Region must be selected');

    if (errors.length > 0) {
      showValidationError('Please fill all fields with valid values: ' + errors.join(', '));
      return false;
    }
    return true;
  }

  function showValidationError(message) {
    const error = document.createElement('div');
    error.className = 'validation-error';
    error.innerHTML = `<i class="fas fa-exclamation-circle"></i><span>${message}</span>`;
    document.body.appendChild(error);
    
    setTimeout(() => error.classList.add('show'), 100);
    setTimeout(() => {
      error.classList.remove('show');
      setTimeout(() => document.body.removeChild(error), 300);
    }, 5000);
  }

  // Add pulse animation keyframes
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.02); }
      100% { transform: scale(1); }
    }
  `;
  document.head.appendChild(style);
});