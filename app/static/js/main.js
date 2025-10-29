document.getElementById('ticket-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const ticket = {
        ticket_subject: document.getElementById('ticket_subject').value,
        ticket_description: document.getElementById('ticket_description').value,
    };

    const response = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(ticket),
    });

    const result = await response.json();

    document.getElementById('prediction-result').innerHTML = `
        <h3>Prediction</h3>
        <p>The predicted category is: <strong>${result.category}</strong></p>
    `;
});
