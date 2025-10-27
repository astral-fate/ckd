document.getElementById("menu-btn").addEventListener("click", function() {
    var sidebar = document.getElementById("sidebar");
    var content = document.querySelector(".content");
    var menuBtn = document.getElementById("menu-btn");
    
    sidebar.classList.toggle("active");
    content.classList.toggle("shifted");
    
    // Hide the menu button when the sidebar is active
    if (sidebar.classList.contains("active")) {
        menuBtn.classList.add("hide");
    } else {
        menuBtn.classList.remove("hide");
    }
});

document.getElementById("close-btn").addEventListener("click", function() {
    var sidebar = document.getElementById("sidebar");
    var content = document.querySelector(".content");
    var menuBtn = document.getElementById("menu-btn");

    sidebar.classList.remove("active");
    content.classList.remove("shifted");
    menuBtn.classList.remove("hide");
});

function showContent(contentId, buttonId) {
    var contents = document.querySelectorAll('.category-content');
    var buttons = document.querySelectorAll('.category-btn');

    contents.forEach(function(content) {
        content.style.display = 'none';
    });

    buttons.forEach(function(button) {
        button.classList.remove('active');
    });

    document.getElementById(contentId).style.display = 'block';
    document.getElementById(buttonId).classList.add('active');
}

document.getElementById("day-btn").addEventListener("click", function() {
    showContent('day-content', 'day-btn');
});

document.getElementById("week-btn").addEventListener("click", function() {
    showContent('week-content', 'week-btn');
});

document.getElementById("month-btn").addEventListener("click", function() {
    showContent('month-content', 'month-btn');
});

// Show the day content by default
showContent('day-content', 'day-btn');


const ckdCtx = document.getElementById('ckdProgressChart').getContext('2d');
new Chart(ckdCtx,
     {
    type: 'line',
    data: {
        labels: ['3/20', '3/21', '3/22', '3/23', '3/24'],
        datasets: [{
            label: 'Progress of the CKD',
            data: [12, 26, 18, 35, 37],
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 40
            }
        },
        plugins: {
            title: {
                display: true,
                text: 'Progress of the CKD',
                font: {
                    size: 18
                }
            }
        }
    }
});

// Meals Chart
const mealsCtx = document.getElementById('mealsChart').getContext('2d');
new Chart(mealsCtx, {
    type: 'bar',
    data: {
        labels: ['Before Breakfast', 'After Breakfast', 'Before Dinner', 'After Dinner'],
        datasets: [
            {
                label: 'GFR',
                data: [15, 20, 40, 150],
                backgroundColor: 'rgba(191, 255, 0, 0.7)',
            },
            {
                label: 'UREA',
                data: [90, 95, 150, 95],
                backgroundColor: 'rgba(255, 192, 203, 0.7)',
            },
            {
                label: 'Keratin',
                data: [10, 12, 15, 25],
                backgroundColor: 'rgba(255, 165, 0, 0.7)',
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 200
            }
        },
        plugins: {
            title: {
                display: true,
                text: 'Before/After the Meals',
                font: {
                    size: 18
                }
            }
        }
    }
});
