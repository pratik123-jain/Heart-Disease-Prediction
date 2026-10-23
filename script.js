
const predictionForm = document.getElementById("predictionForm");
const predictButton = document.getElementById("predictButton");
const resultBox = document.getElementById("resultBox");
const historyBody = document.getElementById("historyBody");

const totalCount = document.getElementById("totalCount");
const higherCount = document.getElementById("higherCount");
const lowerCount = document.getElementById("lowerCount");

// Safely create table cells without inserting user input as HTML
function addCell(row, value, className = "") {
    const cell = document.createElement("td");
    cell.textContent = value ?? "-";

    if (className) {
        cell.className = className;
    }

    row.appendChild(cell);
}

// Load previous predictions
async function loadHistory() {
    try {
        const response = await fetch("/api/dashboard");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not load history.");
        }

        totalCount.textContent = data.total ?? 0;
        higherCount.textContent = data.higher_risk ?? 0;
        lowerCount.textContent = data.lower_risk ?? 0;

        historyBody.replaceChildren();

        const history = data.history || [];

        if (history.length === 0) {
            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 5;
            cell.textContent = "No predictions yet. Submit the form to get started.";

            row.appendChild(cell);
            historyBody.appendChild(row);
            return;
        }

        history.forEach((item) => {
            const row = document.createElement("tr");

            addCell(row, item.patient_name);
            addCell(row, item.age);

            const resultClass =
                item.result === "Higher Risk" ? "risk-high" : "risk-low";

            addCell(row, item.result, resultClass);
            addCell(row, item.created_at);
            addCell(
                row,
                item.probability == null ? "-" : `${item.probability}%`
            );

            historyBody.appendChild(row);
        });
    } catch (error) {
        console.error("History error:", error);

        historyBody.replaceChildren();

        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 5;
        cell.textContent = "Could not load history. Check the Flask backend.";

        row.appendChild(cell);
        historyBody.appendChild(row);
    }
}

// Submit the prediction form
predictionForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const patientName = document.getElementById("patient_name").value.trim();

    if (!patientName) {
        alert("Please enter the patient name.");
        return;
    }

    const fields = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ];

    const data = {
        patient_name: patientName
    };

    for (const field of fields) {
        const input = document.getElementById(field);

        if (input.value === "") {
            alert(`Please fill in ${field}.`);
            input.focus();
            return;
        }

        data[field] = Number(input.value);
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";
    resultBox.style.display = "block";
    resultBox.textContent = "Please wait while the model processes the data...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || "Prediction failed.");
        }

        resultBox.replaceChildren();

        const heading = document.createElement("h3");
        heading.textContent = result.result;

        const details = document.createElement("p");
        details.textContent =
            `Model score: ${result.probability}%`;

        const message = document.createElement("p");
        message.textContent = result.message;

        const disclaimer = document.createElement("p");
        disclaimer.className = "disclaimer";
        disclaimer.textContent =
            "Educational prediction only. This is not a medical diagnosis. " +
            "The model score is not necessarily the probability that you have heart disease.";

        resultBox.append(heading, details, message, disclaimer);

        resultBox.style.background =
            result.prediction === 1 ? "#fff0f1" : "#e9fff2";

        await loadHistory();
    } catch (error) {
        resultBox.replaceChildren();

        const heading = document.createElement("h3");
        heading.textContent = "Unable to predict";

        const message = document.createElement("p");
        message.textContent = error.message;

        resultBox.append(heading, message);
        resultBox.style.background = "#fff0f1";
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "♥ Predict Now";
    }
});

// Clear the old result when the form is reset
predictionForm.addEventListener("reset", function () {
    resultBox.style.display = "none";
    resultBox.replaceChildren();
});

// Load dashboard and history when the page opens
loadHistory();
