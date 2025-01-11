let selectedPlayerId = null;
let performanceChart = null;

document.addEventListener('DOMContentLoaded', function() {
    const playerSearch = document.getElementById('playerSearch');
    const suggestions = document.getElementById('playerSuggestions');
    const analyzePropBtn = document.getElementById('analyzeProp');
    
    // Player search functionality
    let searchTimeout = null;
    playerSearch.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        selectedPlayerId = null;
        
        const query = this.value;
        
        if (query.length < 2) {
            suggestions.innerHTML = '<div class="p-2 text-gray-500">Type at least 2 characters...</div>';
            suggestions.classList.remove('hidden');
            return;
        }
        
        suggestions.innerHTML = '<div class="p-2 text-gray-500">Loading...</div>';
        suggestions.classList.remove('hidden');
        
        searchTimeout = setTimeout(() => {
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
                })
                .catch(error => {
                    console.error('Error:', error);
                    suggestions.innerHTML = '<div class="p-2 text-red-500">Error loading players</div>';
                });
        }, 300);
    });

    // Keyboard navigation for player suggestions
    playerSearch.addEventListener('keydown', function(e) {
        const items = suggestions.querySelectorAll('div:not(.text-gray-500):not(.text-red-500)');
        const active = suggestions.querySelector('.bg-blue-50');
        
        switch(e.key) {
            case 'ArrowDown':
            case 'ArrowUp':
                e.preventDefault();
                handleArrowNavigation(e.key, items, active);
                break;
            case 'Enter':
                if (active) {
                    e.preventDefault();
                    active.click();
                }
                break;
            case 'Escape':
                suggestions.classList.add('hidden');
                break;
        }
    });

    // Analyze prop button handler
    analyzePropBtn.addEventListener('click', async function() {
        if (!selectedPlayerId) {
            alert('Please select a player');
            return;
        }
        
        const propType = document.getElementById('propType').value;
        const line = document.getElementById('lineInput').value;
        const opponentTeamId = document.getElementById('opponentTeam').value;
        
        if (!line) {
            alert('Please enter a line');
            return;
        }
        
        if (!opponentTeamId) {
            alert('Please select an opponent team');
            return;
        }
        
        try {
            analyzePropBtn.disabled = true;
            analyzePropBtn.innerHTML = '<span class="loader"></span> Analyzing...';
            
            // Get player stats
            const statsResponse = await fetch(`/get_player_stats/${selectedPlayerId}`);
            if (!statsResponse.ok) throw new Error('Failed to fetch player stats');
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
                    line: parseFloat(line),
                    opponent_team_id: parseInt(opponentTeamId)
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

    // Close suggestions on click outside
    document.addEventListener('click', function(e) {
        if (!suggestions.contains(e.target) && e.target !== playerSearch) {
            suggestions.classList.add('hidden');
        }
    });
});

function updateResults(analysis, stats, propType, line) {
    try {
        const resultsSection = document.getElementById('results');
        resultsSection.classList.remove('hidden');
        
        // Update main prediction
        updateMainPrediction(analysis);
        
        // Update key metrics
        updateKeyMetrics(analysis, stats);
        
        // Update ML Analysis
        updateMLAnalysis(analysis, stats, propType);
        
        // Update other sections
        updatePlayerContext(analysis.context?.player, stats);
        updateTeamContext(analysis.context?.team);
        updateMatchupAnalysis(analysis.context?.player?.matchup_history);
        updatePerformanceChart(stats, propType, line);
        updateRecentGames(stats, propType, line);
        
    } catch (error) {
        console.error('Error updating results:', error);
        alert('Error displaying results. Please try again.');
    }
}

function updateMainPrediction(analysis) {
    const mainPrediction = document.getElementById('mainPrediction');
    if (!mainPrediction) return;
    
    // Determine color based on recommendation
    let colorClass = '';
    if (analysis.recommendation.includes('STRONG')) {
        colorClass = analysis.recommendation.includes('OVER') ? 'text-green-600' : 'text-red-600';
    } else if (analysis.recommendation.includes('LEAN')) {
        colorClass = analysis.recommendation.includes('OVER') ? 'text-green-500' : 'text-red-500';
    } else {
        colorClass = 'text-gray-600';
    }
    
    // Remove any existing color classes and add the new one
    mainPrediction.className = `text-4xl font-bold mb-4 ${colorClass}`;
    mainPrediction.textContent = `${analysis.recommendation} (${analysis.confidence})`;
}

function updateKeyMetrics(analysis, stats) {
    // Update Predicted Value
    const predictedValue = document.getElementById('predictedValue');
    const edgeValue = document.getElementById('edgeValue');
    if (predictedValue && edgeValue) {
        predictedValue.textContent = analysis.predicted_value.toFixed(1);
        edgeValue.textContent = `${analysis.edge > 0 ? '+' : ''}${(analysis.edge * 100).toFixed(1)}% vs line`;
        edgeValue.className = `text-sm ${analysis.edge > 0 ? 'text-green-600' : 'text-red-600'}`;
    }
    
    // Update Hit Rate
    const hitRate = document.getElementById('hitRate');
    const hitRateDetails = document.getElementById('hitRateDetails');
    if (hitRate && hitRateDetails) {
        hitRate.textContent = `${(analysis.hit_rate * 100).toFixed(1)}%`;
        hitRateDetails.textContent = `${analysis.times_hit} / ${analysis.total_games} games`;
    }
    
    // Update Model Confidence
    const modelConfidence = document.getElementById('modelConfidence');
    if (modelConfidence) {
        modelConfidence.textContent = analysis.confidence;
        modelConfidence.className = `text-2xl font-bold ${
            analysis.confidence === 'HIGH' ? 'text-green-600' :
            analysis.confidence === 'MEDIUM' ? 'text-yellow-600' :
            'text-red-600'
        }`;
    }
}

function updateMLAnalysis(analysis, stats, propType) {
    // Get container elements
    const mainAnalysisText = document.getElementById('mainAnalysisText');
    const classificationConf = document.getElementById('classificationConfidence');
    const regressionPred = document.getElementById('regressionPrediction');
    
    if (!mainAnalysisText || !classificationConf || !regressionPred) {
        console.error('Required ML analysis elements not found');
        return;
    }
    
    const propLabel = getPropTypeLabel(propType);
    
    // Generate analysis text
    let analysisText = `Based on our ML analysis of ${stats.games_played} recent games, `;
    analysisText += `we predict ${propLabel} to go ${analysis.recommendation} with `;
    analysisText += `${analysis.confidence} confidence. The model predicts a value of `;
    analysisText += `${analysis.predicted_value.toFixed(1)} (${analysis.edge > 0 ? '+' : ''}${(analysis.edge * 100).toFixed(1)}% vs line).`;
    
    // Update the DOM elements
    mainAnalysisText.textContent = analysisText;
    classificationConf.textContent = `Over probability: ${(analysis.over_probability * 100).toFixed(1)}%`;
    regressionPred.textContent = `Predicted value: ${analysis.predicted_value.toFixed(1)}`;
    
    // Add color coding based on confidence
    const confidenceColor = analysis.confidence === 'HIGH' ? 'text-green-600' : 
                          analysis.confidence === 'MEDIUM' ? 'text-yellow-600' : 
                          'text-red-600';
    
    mainAnalysisText.className = `text-gray-700 leading-relaxed mb-4 ${confidenceColor}`;
}

function updatePlayerContext(playerContext, stats) {
    const container = document.getElementById('playerContext');
    if (!container) return;
    container.innerHTML = '';
    
    if (playerContext) {
        const items = [
            { label: 'Position', value: playerContext.position },
            { label: 'Recent Form', value: `${(stats.last5_average || 0).toFixed(1)} avg last 5` },
            { label: 'Injury Risk', value: playerContext.injury_history?.injury_risk || 'Low' },
            { label: 'Games Played', value: stats.games_played }
        ];
        
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'context-item';
            div.innerHTML = `
                <span class="text-gray-600">${item.label}</span>
                <span class="font-medium">${item.value}</span>
            `;
            container.appendChild(div);
        });
    }
}

function updateTeamContext(teamContext) {
    const container = document.getElementById('teamContext');
    if (!container) return;
    container.innerHTML = '';
    
    if (teamContext) {
        const items = [
            { label: 'Pace', value: teamContext.pace?.toFixed(1) || 'N/A' },
            { label: 'Offensive Rating', value: teamContext.offensive_rating?.toFixed(1) || 'N/A' },
            { 
                label: 'Injury Impact', 
                value: `${(teamContext.injury_impact * 100).toFixed(1)}%`,
                className: teamContext.injury_impact > 0.15 ? 'text-red-600 font-bold' : ''
            }
        ];
        
        // Add injury details if available
        if (teamContext.injuries && teamContext.injuries.total_players_out > 0) {
            items.push({
                label: 'Players Out',
                value: `${teamContext.injuries.key_players_out} key, ${teamContext.injuries.total_players_out} total`,
                className: 'text-red-600'
            });
            
            // Add individual injuries
            teamContext.injuries.active_injuries.forEach(injury => {
                items.push({
                    label: injury.player_name,
                    value: injury.status,
                    className: 'text-sm text-gray-500 italic'
                });
            });
        }
        
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = `context-item ${item.className || ''}`;
            div.innerHTML = `
                <span class="text-gray-600">${item.label}</span>
                <span class="font-medium">${item.value}</span>
            `;
            container.appendChild(div);
        });
    }
}

function updateMatchupAnalysis(matchupHistory) {
    const container = document.getElementById('matchupAnalysis');
    if (!container) return;
    container.innerHTML = '';
    
    if (matchupHistory) {
        const items = [
            { label: 'VS Team Average', value: matchupHistory.avg_points?.toFixed(1) || 'N/A' },
            { label: 'Success Rate', value: `${(matchupHistory.success_rate * 100).toFixed(1)}%` },
            { label: 'Games Played', value: matchupHistory.games_played || 'N/A' }
        ];
        
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'context-item';
            div.innerHTML = `
                <span class="text-gray-600">${item.label}</span>
                <span class="font-medium">${item.value}</span>
            `;
            container.appendChild(div);
        });
    }
}

function updatePerformanceChart(stats, propType, line) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    const values = propType.includes('_') ? 
        stats.combined_stats[propType]?.values || [] : 
        stats[propType]?.values || [];
    
    const dates = stats.dates?.map(date => {
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
                    text: 'Performance History'
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
}

function updateRecentGames(stats, propType, line) {
    const tbody = document.getElementById('recentGamesBody');
    if (!tbody) return;
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
        'steals': 'Steals',
        'blocks': 'Blocks',
        'turnovers': 'Turnovers',
        'three_pointers': 'Three Pointers Made',
        'pts_reb': 'Points + Rebounds',
        'pts_ast': 'Points + Assists',
        'ast_reb': 'Assists + Rebounds',
        'pts_ast_reb': 'Points + Assists + Rebounds',
        'stl_blk': 'Steals + Blocks',
        'double_double': 'Double Double',
        'triple_double': 'Triple Double'
    };
    return labels[propType] || propType;
}

// Utility functions for keyboard navigation
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