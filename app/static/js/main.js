document.getElementById('ticket-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const ticket = {
        ticket_subject: document.getElementById('ticket_subject').value,
        ticket_description: document.getElementById('ticket_description').value,
        customer_age: parseInt(document.getElementById('customer_age').value),
        customer_gender: document.getElementById('customer_gender').value,
        product_purchased: document.getElementById('product_purchased').value,
        ticket_channel: document.getElementById('ticket_channel').value,
        ticket_priority: document.getElementById('ticket_priority').value,
        ticket_status: document.getElementById('ticket_status').value,
        customer_satisfaction_rating: parseInt(document.getElementById('customer_satisfaction_rating').value),
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
