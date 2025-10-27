document.addEventListener('DOMContentLoaded', function() {
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
                data: [42, 231, 56],
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
                data: [2800, 1400, 900],
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