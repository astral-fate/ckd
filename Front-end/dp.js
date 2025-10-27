document.addEventListener('DOMContentLoaded', function() {
    // Get nutrition values from the model
    const nutrition = {
        protein: parseFloat(document.getElementById('dp-protein').textContent),
        carbs: parseFloat(document.getElementById('dp-carbs').textContent),
        fat: parseFloat(document.getElementById('dp-fat').textContent),
        potassium: parseFloat(document.getElementById('dp-potassium').textContent),
        sodium: parseFloat(document.getElementById('dp-sodium').textContent),
        phosphorus: parseFloat(document.getElementById('dp-phosphorus').textContent)
    };

    // Chart.js code to create the pie charts
    const gramChartCtx = document.getElementById('dp-gram-chart').getContext('2d');
    const mgChartCtx = document.getElementById('dp-mg-chart').getContext('2d');

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            title: {
                display: true,
                font: {
                    size: 14
                }
            }
        }
    };

    new Chart(gramChartCtx, {
        type: 'pie',
        data: {
            labels: ['Protein', 'Carbs', 'Fat'],
            datasets: [{
                data: [nutrition.protein, nutrition.carbs, nutrition.fat],
                backgroundColor: ['#4e79a7', '#283860', '#76b7b2']
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    ...chartOptions.plugins.title,
                    text: 'gram'
                }
            }
        }
    });

    new Chart(mgChartCtx, {
        type: 'pie',
        data: {
            labels: ['Potassium', 'Sodium', 'Phosphorus'],
            datasets: [{
                data: [nutrition.potassium, nutrition.sodium, nutrition.phosphorus],
                backgroundColor: ['#f28e2c', '#e15759', '#ff9da7']
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: {
                    ...chartOptions.plugins.title,
                    text: 'mg'
                }
            }
        }
    });
});