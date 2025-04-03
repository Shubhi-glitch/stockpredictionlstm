document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('stockChart').getContext('2d');

    // Sample stock data
    const stockData = {
        labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        datasets: [{
            label: "NIFTY 50",
            data: [23000, 23500, 22500, 24000, 24500, 23500, 25000, 25500, 24800, 26000, 25500, 27000],
            borderColor: "blue",
            backgroundColor: "rgba(0, 0, 255, 0.1)",
            borderWidth: 2,
            fill: true,
        }]
    };

    // Chart Configuration
    new Chart(ctx, {
        type: "line",
        data: stockData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
});
