document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const resultsContainer = document.getElementById("results");
  const predictedAmount = document.getElementById("predicted-amount");
  const riskLevel = document.getElementById("risk-level");
  const insights = document.getElementById("insights");

  // Enhanced form submission with animations
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

  // Form validation with real-time feedback
  const inputs = form.querySelectorAll('input, select');
  inputs.forEach(input => {
    input.addEventListener('input', validateField);
    input.addEventListener('blur', validateField);
  });

  function validateField(e) {
    const field = e.target;
    const value = field.value;
    let isValid = true;
    let message = '';

    // Validation rules
    if (field.name === 'age') {
      isValid = value >= 18 && value <= 100;
      message = isValid ? '' : 'Age must be between 18 and 100';
    } else if (field.name === 'bmi') {
      isValid = value >= 10 && value <= 60;
      message = isValid ? '' : 'BMI must be between 10 and 60';
    } else if (field.name === 'children') {
      isValid = value >= 0 && value <= 10;
      message = isValid ? '' : 'Children must be between 0 and 10';
    }

    // Update field styling
    if (value && !isValid) {
      field.style.borderColor = 'var(--danger)';
      field.style.boxShadow = '0 0 0 4px rgba(239, 68, 68, 0.1)';
    } else {
      field.style.borderColor = '';
      field.style.boxShadow = '';
    }
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