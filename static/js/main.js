let selectedPlayerId = null;
let performanceChart = null;

document.addEventListener('DOMContentLoaded', function() {
    const playerSearch = document.getElementById('playerSearch');
    const suggestions = document.getElementById('playerSuggestions');
    const analyzePropBtn = document.getElementById('analyzeProp');
    
    // Player search
    let timeout = null;
    playerSearch.addEventListener('input', function() {
        clearTimeout(timeout);
        selectedPlayerId = null;
        
        const query = this.value;
        
        if (query.length < 2) {
            suggestions.innerHTML = '<div class="p-2 text-gray-500">Type at least 2 characters...</div>';
            suggestions.classList.remove('hidden');
            return;
        }
        
        suggestions.innerHTML = '<div class="p-2 text-gray-500">Loading...</div>';
        suggestions.classList.remove('hidden');
        
        timeout = setTimeout(() => {
            fetch(`/search_players?q=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(players => {
                    suggestions.innerHTML = '';
                    if (players.length === 0) {
                        suggestions.innerHTML = '<div class="p-2 text-gray-500">No players found</div>';
                    } else {
                        players.forEach(player => {
                            const div = document.createElement('div');
                            div.className = 'p-2 hover:bg-gray-100 cursor-pointer';
                            div.textContent = player.full_name;
                            div.addEventListener('click', () => {
                                playerSearch.value = player.full_name;
                                selectedPlayerId = player.id;
                                suggestions.classList.add('hidden');
                            });
                            suggestions.appendChild(div);
                        });
                    }
                    suggestions.classList.remove('hidden');
                })
                .catch(error => {
                    console.error('Error:', error);
                    suggestions.innerHTML = '<div class="p-2 text-red-500">Error loading players</div>';
                });
        }, 300);
    });

    // Analyze prop button click
    analyzePropBtn.addEventListener('click', async function() {
        if (!selectedPlayerId) {
            alert('Please select a player');
            return;
        }
        
        const propType = document.getElementById('propType').value;
        const line = document.getElementById('lineInput').value;
        
        if (!line) {
            alert('Please enter a line');
            return;
        }
        
        try {
            analyzePropBtn.disabled = true;
            analyzePropBtn.innerHTML = '<span class="loader"></span> Analyzing...';
            
            // Get player stats
            const statsResponse = await fetch(`/get_player_stats/${selectedPlayerId}`);
            const stats = await statsResponse.json();
            
            // Get prop analysis
            const analysisResponse = await fetch('/analyze_prop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    player_id: selectedPlayerId,
                    prop_type: propType,
                    line: parseFloat(line)
                })
            });
            
            if (!analysisResponse.ok) {
                throw new Error('Analysis failed');
            }
            
            const analysis = await analysisResponse.json();
            
            if (!analysis.success) {
                throw new Error(analysis.error || 'Analysis failed');
            }
            
            updateResults(analysis, stats, propType, line);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing prop: ' + error.message);
        } finally {
            analyzePropBtn.disabled = false;
            analyzePropBtn.innerHTML = 'Analyze Prop';
        }
    });

    document.addEventListener('click', function(e) {
        if (!suggestions.contains(e.target) && e.target !== playerSearch) {
            suggestions.classList.add('hidden');
        }
    });
});

function updateResults(analysis, stats, propType, line) {
    try {
        // Show results section
        generateMLAnalysis(analysis, stats, propType);
        const resultsSection = document.getElementById('results');
        if (!resultsSection) throw new Error('Results section not found');
        
        resultsSection.classList.remove('hidden');
        
        const updateElement = (id, value) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        };
        
        updateElement('hitRate', `${(analysis.probability * 100).toFixed(1)}%`);
        updateElement('average', analysis.average.toFixed(1));
        updateElement('last5Average', analysis.last5_average.toFixed(1));
        
        const recommendationEl = document.getElementById('recommendation');
        if (recommendationEl) {
            recommendationEl.textContent = `${analysis.recommendation} (${analysis.confidence})`;
            recommendationEl.className = `text-2xl font-bold ${getConfidenceColor(analysis.confidence)}`;
        }
        
        updateElement('timesOver', `${analysis.times_hit} / ${analysis.total_games}`);
        updateElement('timesUnder', `${analysis.total_games - analysis.times_hit} / ${analysis.total_games}`);
        
        if (analysis.trend) {
            updateElement('trendDirection', analysis.trend.direction);
            updateElement('valueVsLine', 
                `${(analysis.edge * 100).toFixed(1)}% ${analysis.edge > 0 ? 'above' : 'below'}`);
            updateElement('edge', `${(analysis.edge * 100).toFixed(1)}%`);
        }
        
        updatePerformanceChart(stats, propType, line);
        updateRecentGames(stats, propType, line);
        
    } catch (error) {
        console.error('Error updating results:', error);
        alert('Error displaying results. Please try again.');
    }
}

function handleArrowNavigation(key, items, active) {
    if (items.length === 0) return;
    
    let nextIndex;
    if (!active) {
        nextIndex = key === 'ArrowDown' ? 0 : items.length - 1;
    } else {
        const currentIndex = Array.from(items).indexOf(active);
        active.classList.remove('bg-blue-50');
        
        if (key === 'ArrowDown') {
            nextIndex = currentIndex + 1 >= items.length ? 0 : currentIndex + 1;
        } else {
            nextIndex = currentIndex - 1 < 0 ? items.length - 1 : currentIndex - 1;
        }
    }
    
    items[nextIndex].classList.add('bg-blue-50');
    items[nextIndex].scrollIntoView({ block: 'nearest' });
}

function getConfidenceColor(confidence) {
    switch (confidence) {
        case 'HIGH':
            return 'text-green-600';
        case 'MEDIUM':
            return 'text-yellow-600';
        case 'LOW':
            return 'text-red-600';
        default:
            return '';
    }
}

function updatePerformanceChart(stats, propType, line) {
    try {
        const ctx = document.getElementById('performanceChart');
        
        if (performanceChart) {
            performanceChart.destroy();
        }
        
        const values = propType.includes('_') ? 
            stats.combined_stats[propType]?.values || [] : 
            stats[propType]?.values || [];
        
        let dates = stats.dates?.map(date => {
            const d = new Date(date);
            return d.toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric'
            });
        }) || [];
        
        const reversedValues = [...values].reverse();
        const reversedDates = [...dates].reverse();
        
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: reversedDates,
                datasets: [
                    {
                        label: 'Actual',
                        data: reversedValues,
                        borderColor: 'rgb(59, 130, 246)',
                        tension: 0.1,
                        fill: false
                    },
                    {
                        label: 'Line',
                        data: Array(reversedDates.length).fill(line),
                        borderColor: 'rgb(239, 68, 68)',
                        borderDash: [5, 5],
                        tension: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Last 20 Games Performance'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: getPropTypeLabel(propType)
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Game Date'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error updating performance chart:', error);
    }
 }

function generateMLAnalysis(analysis, stats, propType) {
    const propLabel = getPropTypeLabel(propType);
    const averagePerf = analysis.average.toFixed(1);
    const predictedValue = analysis.predicted_value?.toFixed(1) || averagePerf;
    const recentForm = analysis.last5_average.toFixed(1);
    const trend = analysis.trend && analysis.trend.direction ? 
        analysis.trend.direction.toLowerCase() : 'stable';

    let reasoning = `Based on our ML analysis of ${stats.games_played} recent games, `;
    
    if (analysis.recommendation === 'OVER') {
        reasoning += `we predict ${propLabel} to go OVER with ${(analysis.over_probability * 100).toFixed(1)}% confidence. `;
    } else if (analysis.recommendation === 'UNDER') {
        reasoning += `we predict ${propLabel} to go UNDER with ${((1 - analysis.over_probability) * 100).toFixed(1)}% confidence. `;
    } else {
        reasoning += 'this prop shows no clear edge. ';
    }

    reasoning += `The player's performance shows a ${trend} trend, averaging ${averagePerf} with recent form at ${recentForm}. `;
    reasoning += `Our model predicts a value of ${predictedValue}, `;
    
    if (analysis.edge > 0) {
        reasoning += `suggesting a ${(analysis.edge * 100).toFixed(1)}% edge over the line.`;
    } else {
        reasoning += `indicating the line may be ${Math.abs(analysis.edge * 100).toFixed(1)}% too high.`;
    }

    document.querySelector('#mlAnalysis p').textContent = reasoning;
    document.getElementById('classificationConfidence').textContent = 
        `Over probability: ${(analysis.over_probability * 100).toFixed(1)}%`;
    document.getElementById('regressionPrediction').textContent = 
        `Predicted value: ${predictedValue} (${analysis.edge > 0 ? '+' : ''}${(analysis.edge * 100).toFixed(1)}% vs line)`;
}

function updateRecentGames(stats, propType, line) {
    const tbody = document.getElementById('recentGamesBody');
    tbody.innerHTML = '';
    
    let values;
    if (propType === 'three_pointers') {
        values = stats.three_pointers?.values || [];
    } else if (propType === 'double_double' || propType === 'triple_double') {
        values = stats[propType]?.values || [];
    } else if (propType.includes('_')) {
        values = stats.combined_stats[propType]?.values || [];
    } else {
        values = stats[propType]?.values || [];
    }
    
    values.forEach((value, index) => {
        const row = document.createElement('tr');
        row.className = value > line ? 'bg-green-50' : 'bg-red-50';
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap">${new Date(stats.dates[index]).toLocaleDateString()}</td>
            <td class="px-6 py-4 whitespace-nowrap">${stats.matchups[index]}</td>
            <td class="px-6 py-4 whitespace-nowrap font-medium ${value > line ? 'text-green-600' : 'text-red-600'}">${value.toFixed(1)}</td>
            <td class="px-6 py-4 whitespace-nowrap">${value > line ? 'OVER' : 'UNDER'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function getPropTypeLabel(propType) {
    const labels = {
        'points': 'Points',
        'assists': 'Assists',
        'rebounds': 'Rebounds',
        'pts_reb': 'Points + Rebounds',
        'pts_ast': 'Points + Assists',
        'ast_reb': 'Assists + Rebounds',
        'pts_ast_reb': 'Points + Assists + Rebounds'
    };
    return labels[propType] || propType;
}