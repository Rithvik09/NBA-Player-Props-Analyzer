let selectedPlayerId = null;
let performanceChart = null;

async function autoFillOpponent(playerId) {
    const opponentSelect = document.getElementById('opponentTeam');
    const locationNote   = document.getElementById('locationNote');

    if (locationNote) {
        locationNote.textContent = "Detecting today's game...";
        locationNote.className = 'text-xs text-gray-400 mt-1';
    }

    try {
        const res  = await fetch(`/player_game_info/${playerId}`);
        if (!res.ok) throw new Error(`server returned ${res.status}`);
        const data = await res.json();

        if (data.opponent_team_id) {
                    opponentSelect.value = data.opponent_team_id;

            if (data.is_home !== null) {
                setHomeAway(null); // keep on Auto — backend will confirm
                if (locationNote) {
                    locationNote.textContent = data.is_home
                        ? '🏠 Auto-detected: Home game today'
                        : '✈️ Auto-detected: Away game today';
                    locationNote.className = 'text-xs text-blue-500 mt-1 font-medium';
                }
            }
        } else {
            if (locationNote) {
                locationNote.textContent = 'No game found today — select opponent manually';
                locationNote.className = 'text-xs text-yellow-500 mt-1';
            }
        }
    } catch (e) {
        console.error('autoFillOpponent error:', e);
        if (locationNote) {
            locationNote.textContent = "Couldn't detect today's game";
            locationNote.className = 'text-xs text-red-400 mt-1';
        }
    }
}

function setHomeAway(isHome) {
    document.getElementById('isHome').value = isHome === null ? '' : (isHome ? 'true' : 'false');
    const autoBtn = document.getElementById('autoBtn');
    const homeBtn = document.getElementById('homeBtn');
    const awayBtn = document.getElementById('awayBtn');
    const note    = document.getElementById('locationNote');
    if (autoBtn) autoBtn.className = isHome === null  ? 'loc-btn loc-active' : 'loc-btn';
    if (homeBtn) homeBtn.className = isHome === true  ? 'loc-btn loc-active' : 'loc-btn';
    if (awayBtn) awayBtn.className = isHome === false ? 'loc-btn loc-active' : 'loc-btn';
    if (note) note.textContent = isHome === null
        ? "Auto-detects from today's schedule"
        : isHome ? 'Manually set to Home' : 'Manually set to Away';
}

document.addEventListener('DOMContentLoaded', function() {
    const playerSearch = document.getElementById('playerSearch');
    const suggestions = document.getElementById('playerSuggestions');
    const analyzePropBtn = document.getElementById('analyzeProp');
    
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
                .then(response => {
                    if (!response.ok) throw new Error(`search failed: ${response.status}`);
                    return response.json();
                })
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
                                autoFillOpponent(player.id);
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
            
            const statsResponse = await fetch(`/get_player_stats/${selectedPlayerId}`);
            if (!statsResponse.ok) throw new Error('Failed to fetch player stats');
            const stats = await statsResponse.json();
            
            const analysisResponse = await fetch('/analyze_prop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    player_id: selectedPlayerId,
                    prop_type: propType,
                    line: parseFloat(line),
                    opponent_team_id: parseInt(opponentTeamId),
                    is_home: document.getElementById('isHome').value === '' ? null :
                             document.getElementById('isHome').value === 'true'
                })
            });
            
            if (!analysisResponse.ok) {
                throw new Error('Analysis failed');
            }
            
            const analysis = await analysisResponse.json();
            
            if (!analysis.success) {
                throw new Error(analysis.error || 'Analysis failed');
            }
            
            updateResults(analysis, stats, propType, parseFloat(line));
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing prop: ' + error.message);
        } finally {
            analyzePropBtn.disabled = false;
            analyzePropBtn.innerHTML = '<span class="analyze-btn-inner">⚡ Analyze Prop</span>';
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
        const resultsSection = document.getElementById('results');
        resultsSection.classList.remove('hidden');

        const locationNote = document.getElementById('locationNote');
        if (locationNote && analysis.location_detected) {
            locationNote.textContent = analysis.is_home
                ? '🏠 Auto-detected: Home game'
                : '✈️ Auto-detected: Away game';
            locationNote.className = 'text-xs text-blue-500 mt-1 font-medium';
        }
        
        updateKeyMetrics(analysis, stats);
        
        const mainConclusion = document.getElementById('mainConclusion');
        if (mainConclusion) {
            const colorClass = analysis.recommendation.includes('OVER') ? 'text-green-600' : 
                             analysis.recommendation.includes('UNDER') ? 'text-red-600' : 
                             'text-gray-600';
            mainConclusion.className = `text-4xl font-bold mb-4 ${colorClass}`;
            mainConclusion.textContent = `${analysis.recommendation} (${analysis.confidence})`;
        }

        updateMLAnalysis(analysis, stats, propType);
        updatePlayerContext(analysis.context?.player, stats, propType, analysis);
        updateTeamContext(analysis.context?.team);
        updateMatchupAnalysis(
            analysis.context?.player?.matchup_history,
            analysis.context?.player?.position_matchup
        );
        updatePerformanceChart(stats, propType, line);
        updateRecentGames(stats, propType, line);
        
    } catch (error) {
        console.error('Error updating results:', error);
        alert('Error displaying results. Please try again.');
    }
}


function updateKeyMetrics(analysis, stats) {
    const predictedValue = document.getElementById('predictedValue');
    const edgeValue = document.getElementById('edgeValue');
    if (predictedValue && edgeValue) {
        predictedValue.textContent = analysis.predicted_value.toFixed(1);
        edgeValue.textContent = `${analysis.edge > 0 ? '+' : ''}${(analysis.edge * 100).toFixed(1)}% vs line`;
        edgeValue.className = `text-sm ${analysis.edge > 0 ? 'text-green-600' : 'text-red-600'}`;
    }
    
    const hitRate = document.getElementById('hitRate');
    const hitRateDetails = document.getElementById('hitRateDetails');
    if (hitRate && hitRateDetails) {
        hitRate.textContent = `${(analysis.hit_rate * 100).toFixed(1)}%`;
        hitRateDetails.textContent = `${analysis.times_hit} / ${analysis.total_games} games`;
    }
    
    const modelConfidence = document.getElementById('modelConfidence');
    if (modelConfidence) {
        modelConfidence.textContent = analysis.confidence;
        modelConfidence.className = `text-3xl font-bold ${
            analysis.confidence === 'HIGH' ? 'text-green-600' :
            analysis.confidence === 'MEDIUM' ? 'text-yellow-600' :
            'text-red-600'
        }`;
    }
}

function updateMLAnalysis(analysis, stats, propType) {
    const mainAnalysisText = document.getElementById('mainAnalysisText');
    const classificationConf = document.getElementById('classificationConfidence');
    const regressionPred = document.getElementById('regressionPrediction');

    if (!mainAnalysisText || !classificationConf || !regressionPred) {
        console.error('Required ML analysis elements not found');
        return;
    }

    const propLabel = getPropTypeLabel(propType);
    const player   = analysis.context?.player  || {};
    const team     = analysis.context?.team    || {};
    const opponent = analysis.context?.opponent || {};
    const matchup  = player.matchup_history    || null;
    const posDef   = player.position_matchup   || null;

    // combined props live under stats.combined_stats
    const propStats = propType && propType in stats
        ? (stats[propType] || {})
        : (propType && stats.combined_stats && propType in stats.combined_stats)
            ? (stats.combined_stats[propType] || {})
            : (stats['points'] || {});
    const seasonAvg  = propStats.avg       ?? null;
    const last5Avg   = propStats.last5_avg ?? null;
    const trend      = analysis.trend      || {};

    // each factor: { label, summary, strength ('strong'|'moderate'|'weak'), bullish }
    const factors = [];

    // home/away split
    const isHome = analysis.is_home ?? null;  // null = location unknown
    const locationAvgFactor = isHome === true ? propStats.home_avg : isHome === false ? propStats.away_avg : null;
    const locationGamesFactor = isHome === true ? propStats.home_games : isHome === false ? propStats.away_games : null;
    if (isHome !== null && locationAvgFactor != null && seasonAvg != null && (locationGamesFactor ?? 0) >= 5) {
        const diff = locationAvgFactor - seasonAvg;
        const pct  = seasonAvg > 0 ? (diff / seasonAvg) * 100 : 0;
        const isBullish = diff > 0;
        factors.push({
            label: isHome ? 'Home Advantage' : 'Away Performance',
            summary: isBullish
                ? `Averages ${locationAvgFactor.toFixed(1)} ${propLabel.toLowerCase()} ${isHome ? 'at home' : 'on the road'} vs ${seasonAvg.toFixed(1)} overall (+${Math.abs(pct).toFixed(0)}%)`
                : `Averages ${locationAvgFactor.toFixed(1)} ${propLabel.toLowerCase()} ${isHome ? 'at home' : 'on the road'} vs ${seasonAvg.toFixed(1)} overall (-${Math.abs(pct).toFixed(0)}%)`,
            strength: Math.abs(pct) > 15 ? 'strong' : Math.abs(pct) > 7 ? 'moderate' : 'weak',
            bullish: isBullish
        });
    }

    // recent trend
    if (trend.direction) {
        const dir = trend.direction;
        const slope = trend.slope ?? 0;
        const isBullish = dir === 'Increasing';
        factors.push({
            label: 'Recent Trend',
            summary: dir === 'Increasing'
                ? `Player is on an upward trend (+${Math.abs(slope).toFixed(2)} per game over last 5)`
                : dir === 'Decreasing'
                    ? `Player is trending downward (${Math.abs(slope).toFixed(2)} drop per game over last 5)`
                    : `Player has been stable over the last 5 games`,
            strength: Math.abs(slope) > 2 ? 'strong' : Math.abs(slope) > 0.5 ? 'moderate' : 'weak',
            bullish: isBullish
        });
    }

    // hot/cold: last 5 vs season avg
    if (seasonAvg != null && last5Avg != null) {
        const diff = last5Avg - seasonAvg;
        const pct  = seasonAvg > 0 ? (diff / seasonAvg) * 100 : 0;
        const isBullish = diff > 0;
        factors.push({
            label: 'Recent Form vs Season',
            summary: isBullish
                ? `Running ${Math.abs(pct).toFixed(0)}% above season average over last 5 games (${last5Avg.toFixed(1)} vs ${seasonAvg.toFixed(1)} avg)`
                : `Running ${Math.abs(pct).toFixed(0)}% below season average over last 5 games (${last5Avg.toFixed(1)} vs ${seasonAvg.toFixed(1)} avg)`,
            strength: Math.abs(pct) > 20 ? 'strong' : Math.abs(pct) > 10 ? 'moderate' : 'weak',
            bullish: isBullish
        });
    }

    // historical hit rate
    if (analysis.hit_rate != null) {
        const hr = analysis.hit_rate;
        const isBullish = hr > 0.5;
        factors.push({
            label: 'Historical Hit Rate',
            summary: `Gone OVER this line in ${(hr * 100).toFixed(0)}% of recent games (${analysis.times_hit}/${analysis.total_games})`,
            strength: hr > 0.70 || hr < 0.30 ? 'strong' : hr > 0.60 || hr < 0.40 ? 'moderate' : 'weak',
            bullish: isBullish
        });
    }

    // head-to-head history vs this opponent
    if (matchup && matchup.games_played > 0) {
        let matchupAvg = matchup.avg_points ?? 0;
        if (propType === 'pts_reb')      matchupAvg = (matchup.avg_points ?? 0) + (matchup.avg_rebounds ?? 0);
        else if (propType === 'pts_ast') matchupAvg = (matchup.avg_points ?? 0) + (matchup.avg_assists  ?? 0);
        else if (propType === 'ast_reb') matchupAvg = (matchup.avg_assists ?? 0) + (matchup.avg_rebounds ?? 0);
        else if (propType === 'pts_ast_reb') matchupAvg = (matchup.avg_points ?? 0) + (matchup.avg_assists ?? 0) + (matchup.avg_rebounds ?? 0);
        else if (propType === 'assists')  matchupAvg = matchup.avg_assists  ?? 0;
        else if (propType === 'rebounds') matchupAvg = matchup.avg_rebounds ?? 0;

        const successR = matchup.success_rate ?? 0.5;
        const isBullish = successR > 0.5;
        factors.push({
            label: 'Head-to-Head History',
            summary: `Averaging ${matchupAvg.toFixed(1)} ${propLabel.toLowerCase()} in ${matchup.games_played} prior matchups vs this team (${(successR * 100).toFixed(0)}% win rate)`,
            strength: matchup.games_played >= 5 ? (Math.abs(successR - 0.5) > 0.2 ? 'strong' : 'moderate') : 'weak',
            bullish: isBullish
        });
    }

    // 5. Opponent positional defense
    if (posDef && posDef.defensive_rating != null) {
        const defRtg   = posDef.defensive_rating;
        const ptsAllowed = posDef.pts_allowed_per_game ?? null;
        const isBullish = defRtg > 110;  // high def rating = worse defense
        factors.push({
            label: 'Opponent Positional Defense',
            summary: ptsAllowed != null
                ? `Opponent allows ${ptsAllowed.toFixed(1)} pts/game to this position (def rating: ${defRtg.toFixed(0)})`
                : `Opponent defensive rating vs position: ${defRtg.toFixed(0)}`,
            strength: defRtg > 115 || defRtg < 105 ? 'strong' : 'moderate',
            bullish: isBullish
        });
    }

    // opponent injuries
    if (opponent.injury_impact != null && opponent.injury_impact > 0.05) {
        const impact = opponent.injury_impact;
        factors.push({
            label: 'Opponent Injuries',
            summary: `Opponent missing key personnel (injury impact: ${(impact * 100).toFixed(0)}%) — weakened defense`,
            strength: impact > 0.3 ? 'strong' : impact > 0.15 ? 'moderate' : 'weak',
            bullish: true
        });
    }

    // player's own team injuries (hurts usage)
    if (team.injury_impact != null && team.injury_impact > 0.05) {
        const impact = team.injury_impact;
        factors.push({
            label: 'Team Injuries',
            summary: `Player\'s own team is short-handed (injury impact: ${(impact * 100).toFixed(0)}%) — may affect usage/pace`,
            strength: impact > 0.3 ? 'strong' : impact > 0.15 ? 'moderate' : 'weak',
            bullish: false
        });
    }

    // rest days / fatigue
    if (team.rest_days != null) {
        const rest = team.rest_days;
        const isBullish = rest >= 2;
        if (rest <= 1 || rest >= 3) {
            factors.push({
                label: 'Rest & Fatigue',
                summary: rest === 0
                    ? `Back-to-back game — fatigue is a significant concern`
                    : rest === 1
                        ? `Only 1 day of rest — slight fatigue risk`
                        : rest >= 4
                            ? `${rest} days of rest — well-rested, could be fresh`
                            : `${rest} days rest — normal schedule`,
                strength: rest === 0 ? 'strong' : rest === 1 || rest >= 4 ? 'moderate' : 'weak',
                bullish: isBullish
            });
        }
    }

    const strengthOrder = { strong: 0, moderate: 1, weak: 2 };
    factors.sort((a, b) => strengthOrder[a.strength] - strengthOrder[b.strength]);

    const edgePct = (analysis.edge * 100).toFixed(1);
    const edgeStr = `${analysis.edge > 0 ? '+' : ''}${edgePct}%`;

    let narrative = `Model predicts ${propLabel} at ${analysis.predicted_value.toFixed(1)} (${edgeStr} vs line) `;
    narrative += `with ${(analysis.over_probability * 100).toFixed(1)}% over probability. `;

    if (factors.length === 0) {
        narrative += `Recommendation: ${analysis.recommendation} (${analysis.confidence} confidence).`;
    } else {
        const bullish  = factors.filter(f => f.bullish);
        const bearish  = factors.filter(f => !f.bullish);
        const topFor   = bullish.slice(0, 2);
        const topAgainst = bearish.slice(0, 1);

        if (topFor.length > 0) {
            narrative += `Key factors supporting the OVER: `;
            narrative += topFor.map(f => `${f.label} (${f.strength}) — ${f.summary}`).join('; ');
            narrative += '. ';
        }
        if (topAgainst.length > 0) {
            narrative += `Main risk: `;
            narrative += topAgainst.map(f => `${f.label} (${f.strength}) — ${f.summary}`).join('; ');
            narrative += '. ';
        }
        narrative += `Overall: ${analysis.recommendation} (${analysis.confidence} confidence).`;
    }

    mainAnalysisText.textContent = narrative;

    const teamInjuries = team.injuries?.active_injuries || [];
    const oppInjuries  = opponent.injuries?.active_injuries || [];

    const renderInjuryList = (injuryList, side) => {
        if (injuryList.length === 0) return `<span class="text-gray-400 text-xs">No reported injuries</span>`;
        return injuryList.map(inj => {
            const isKey = inj.impact_score != null && inj.impact_score > 0.15;
            const color = side === 'team' ? 'text-red-600' : 'text-green-600';
            return `<span class="${color} text-xs">${isKey ? '🔴' : '🟡'} <strong>${inj.player_name}</strong> — ${inj.injury} (${inj.status}${inj.expected_return ? ', ret: ' + inj.expected_return : ''})</span>`;
        }).join('<br>');
    };

    // fill in the Prediction Model / Value Analysis cards
    classificationConf.innerHTML = `
        Over probability: <strong>${(analysis.over_probability * 100).toFixed(1)}%</strong><br>
        ${factors.slice(0, 3).map(f =>
            `<span class="${f.bullish ? 'text-green-600' : 'text-red-600'}">
                ${f.bullish ? '▲' : '▼'} ${f.label}: ${f.summary}
            </span>`
        ).join('<br>')}
        <hr class="my-2 border-gray-200">
        <span class="text-xs font-semibold text-gray-500 uppercase tracking-wide">Opponent Injuries (${oppInjuries.length})</span><br>
        ${renderInjuryList(oppInjuries, 'opp')}
    `;
    regressionPred.innerHTML = `
        Predicted value: <strong>${analysis.predicted_value.toFixed(1)}</strong><br>
        Edge vs line: <strong class="${analysis.edge > 0 ? 'text-green-600' : 'text-red-600'}">${edgeStr}</strong><br>
        ${factors.slice(3).map(f =>
            `<span class="${f.bullish ? 'text-green-600' : 'text-red-600'}">
                ${f.bullish ? '▲' : '▼'} ${f.label}: ${f.summary}
            </span>`
        ).join('<br>')}
        <hr class="my-2 border-gray-200">
        <span class="text-xs font-semibold text-gray-500 uppercase tracking-wide">Team Injuries (${teamInjuries.length})</span><br>
        ${renderInjuryList(teamInjuries, 'team')}
    `;

    const confidenceColor = analysis.confidence === 'HIGH' ? 'text-green-600' :
                            analysis.confidence === 'MEDIUM' ? 'text-yellow-600' :
                            'text-red-600';
    mainAnalysisText.className = `analysis-text mb-4 ${confidenceColor}`;
}

function updatePlayerContext(playerContext, stats, propType, analysis) {
    const container = document.getElementById('playerContext');
    if (!container) return;
    container.innerHTML = '';

    if (playerContext && stats) {
        const propStats = propType && propType in stats
            ? (stats[propType] || {})
            : (propType && stats.combined_stats && propType in stats.combined_stats)
                ? (stats.combined_stats[propType] || {})
                : (stats['points'] || {});
        
        const isHome = analysis?.is_home ?? null;  // null = unknown/auto
        const locationLabel = isHome === true ? '🏠 Home Avg' : isHome === false ? '✈️ Away Avg' : '📍 Location Avg';
        const locationAvg = isHome === true ? propStats.home_avg : isHome === false ? propStats.away_avg : propStats.avg;
        const locationGames = isHome === true ? propStats.home_games : isHome === false ? propStats.away_games : null;

        const items = [
            { label: 'Position', value: playerContext.position || 'N/A' },
            {
                label: 'Season Average',
                value: propStats.avg != null ? `${propStats.avg.toFixed(1)}` : 'N/A'
            },
            {
                label: locationLabel,
                value: locationAvg != null ? `${locationAvg.toFixed(1)} (${locationGames} games)` : 'N/A'
            },
            {
                label: 'Last 5 Games',
                value: propStats.last5_avg != null ? `${propStats.last5_avg.toFixed(1)}` : 'N/A'
            },
            { label: 'Games Played', value: stats.games_played || 'N/A' },
            { label: 'Trend', value: propStats?.direction || 'Stable' }
        ];
        
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'flex justify-between items-center py-2';
            div.innerHTML = `
                <span class="text-gray-600">${item.label}</span>
                <span class="font-medium">${item.value}</span>
            `;
            container.appendChild(div);
        });
    } else {
        container.innerHTML = '<div class="text-gray-500 text-center py-4">No player data available</div>';
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
        
        if (teamContext.injuries && teamContext.injuries.total_players_out > 0) {
            items.push({
                label: 'Players Out',
                value: `${teamContext.injuries.key_players_out} key, ${teamContext.injuries.total_players_out} total`,
                className: 'text-red-600'
            });
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

function updateMatchupAnalysis(matchupHistory, positionMatchup) {
    const container = document.getElementById('matchupAnalysis');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (matchupHistory || positionMatchup) {
        const items = [];
        
        if (matchupHistory) {
            items.push(
                {
                    label: 'VS Team Average',
                    value: matchupHistory.avg_points != null ? `${matchupHistory.avg_points.toFixed(1)}` : 'N/A'
                },
                {
                    label: 'Previous Matchups',
                    value: matchupHistory.games_played ?? 0
                },
                {
                    label: 'Success Rate',
                    value: matchupHistory.success_rate != null ?
                           `${(matchupHistory.success_rate * 100).toFixed(1)}%` : 'N/A'
                }
            );
        }
        
        if (positionMatchup) {
            items.push(
                {
                    label: 'Position Defense',
                    value: positionMatchup.defensive_rating != null ?
                           positionMatchup.defensive_rating.toFixed(1) : 'N/A'
                },
                {
                    label: 'Points Allowed',
                    value: positionMatchup.pts_allowed_per_game != null ?
                           positionMatchup.pts_allowed_per_game.toFixed(1) : 'N/A'
                }
            );
        }
        
        if (items.length > 0) {
            items.forEach(item => {
                const div = document.createElement('div');
                div.className = 'flex justify-between items-center py-2';
                div.innerHTML = `
                    <span class="text-gray-600">${item.label}</span>
                    <span class="font-medium">${item.value}</span>
                `;
                container.appendChild(div);
            });
        } else {
            container.innerHTML = '<div class="text-gray-500 text-center py-4">No matchup data available</div>';
        }
    } else {
        container.innerHTML = '<div class="text-gray-500 text-center py-4">No matchup data available</div>';
    }
}

function updatePerformanceChart(stats, propType, line) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    let values;
    if (propType === 'three_pointers') {
        values = stats.three_pointers?.values || [];
    } else if (propType === 'double_double' || propType === 'triple_double') {
        values = stats[propType]?.values || [];
    } else if (propType.includes('_')) {
        values = stats.combined_stats?.[propType]?.values || [];
    } else {
        values = stats[propType]?.values || [];
    }
    
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
        values = stats.combined_stats?.[propType]?.values || [];
    } else {
        values = stats[propType]?.values || [];
    }
    
    const dates    = stats.dates    || [];
    const matchups = stats.matchups || [];
    values.forEach((value, index) => {
        const row = document.createElement('tr');
        row.className = value > line ? 'bg-green-50' : 'bg-red-50';
        const dateStr    = dates[index]    ? new Date(dates[index]).toLocaleDateString() : '—';
        const matchupStr = matchups[index] ?? '—';
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap">${dateStr}</td>
            <td class="px-6 py-4 whitespace-nowrap">${matchupStr}</td>
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

// ══════════════════════════════════════════════════════
// TAB NAVIGATION
// ══════════════════════════════════════════════════════
function showTab(tab) {
    ['analyzer', 'logs', 'accuracy', 'bias'].forEach(t => {
        document.getElementById(`tab-content-${t}`).classList.toggle('hidden', t !== tab);
        const btn = document.getElementById(`tab-${t}`);
        if (btn) {
            btn.className = t === tab ? 'tab-btn tab-active' : 'tab-btn';
        }
    });
    if (tab === 'logs') loadLogs();
    if (tab === 'accuracy') loadAccuracy();
    if (tab === 'bias') loadBias();
}

// ══════════════════════════════════════════════════════
// PREDICTION LOGS
// ══════════════════════════════════════════════════════
async function triggerAutoGrade() {
    const status = document.getElementById('autoGradeStatus');
    if (status) status.textContent = 'Fetching results...';
    try {
        const agRes = await fetch('/logs/auto-grade', { method: 'POST' });
        if (!agRes.ok) throw new Error(`auto-grade failed: ${agRes.status}`);
        // give the background thread a moment to finish then refresh
        setTimeout(async () => {
            await loadLogs();
            if (status) status.textContent = 'Done ✓';
            setTimeout(() => { if (status) status.textContent = ''; }, 3000);
        }, 4000);
    } catch (e) {
        if (status) status.textContent = 'Error: ' + e.message;
    }
}

async function loadLogs() {
    const container = document.getElementById('logsContainer');
    container.innerHTML = '<p class="text-gray-400 text-center py-8">Loading...</p>';
    try {
        const res = await fetch('/logs?limit=100');
        if (!res.ok) throw new Error(`failed to load logs: ${res.status}`);
        const logs = await res.json();
        if (!logs.length) {
            container.innerHTML = '<p class="text-gray-400 text-center py-8">No predictions logged yet. Run an analysis first.</p>';
            return;
        }
        renderLogsTable(logs, container);
    } catch (e) {
        container.innerHTML = `<p class="text-red-500 text-center py-8">Error loading logs: ${e.message}</p>`;
    }
}

function renderLogsTable(logs, container) {
    const confColor = c => c === 'HIGH' ? 'text-green-600 font-bold' : c === 'MEDIUM' ? 'text-yellow-600' : 'text-gray-400';
    const recColor  = r => r.includes('OVER') ? 'text-green-600' : r.includes('UNDER') ? 'text-red-600' : 'text-gray-400';
    const correctBadge = row => {
        if (row.correct === 1) return `<span class="text-xs bg-green-100 text-green-700 px-2 py-1 rounded font-bold">✓ Correct</span>`;
        if (row.correct === 0) return `<span class="text-xs bg-red-100 text-red-700 px-2 py-1 rounded font-bold">✗ Wrong</span>`;
        // null = PASS or not yet graded
        if (row.actual_result != null) {
            return `<span class="text-xs bg-gray-100 text-gray-500 px-2 py-1 rounded">PASS</span>`;
        }
        return `<button onclick="openModal(${row.id})"
            class="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded">+ Result</button>`;
    };

    container.innerHTML = `
        <table class="min-w-full text-sm">
            <thead class="bg-gray-50 text-xs uppercase text-gray-500">
                <tr>
                    <th class="px-3 py-2 text-left">Date</th>
                    <th class="px-3 py-2 text-left">Player</th>
                    <th class="px-3 py-2 text-left">Prop</th>
                    <th class="px-3 py-2 text-right">Line</th>
                    <th class="px-3 py-2 text-right">Predicted</th>
                    <th class="px-3 py-2 text-left">Rec</th>
                    <th class="px-3 py-2 text-left">Conf</th>
                    <th class="px-3 py-2 text-right">Actual</th>
                    <th class="px-3 py-2 text-center">Result</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-gray-100">
                ${logs.map(row => `
                    <tr class="hover:bg-gray-50 ${row.correct === 1 ? 'bg-green-50' : row.correct === 0 ? 'bg-red-50' : row.actual_result != null ? 'bg-gray-50' : ''}">
                        <td class="px-3 py-2 whitespace-nowrap text-gray-400">${row.timestamp.slice(0,10)}</td>
                        <td class="px-3 py-2 font-medium">${row.player_name || row.player_id}</td>
                        <td class="px-3 py-2">${getPropTypeLabel(row.prop_type)}</td>
                        <td class="px-3 py-2 text-right">${row.line}</td>
                        <td class="px-3 py-2 text-right">${row.predicted_value != null ? row.predicted_value.toFixed(1) : '—'}</td>
                        <td class="px-3 py-2 ${recColor(row.recommendation || '')}">${row.recommendation || '—'}</td>
                        <td class="px-3 py-2 ${confColor(row.confidence || '')}">${row.confidence || '—'}</td>
                        <td class="px-3 py-2 text-right">${row.actual_result != null ? row.actual_result : '—'}</td>
                        <td class="px-3 py-2 text-center">${correctBadge(row)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// ══════════════════════════════════════════════════════
// RESULT MODAL
// ══════════════════════════════════════════════════════
function openModal(logId) {
    document.getElementById('modalLogId').value = logId;
    document.getElementById('modalActualResult').value = '';
    document.getElementById('modalNotes').value = '';
    document.getElementById('resultModal').classList.remove('hidden');
    document.getElementById('modalActualResult').focus();
}

function closeModal() {
    document.getElementById('resultModal').classList.add('hidden');
}

async function submitResult() {
    const logId = document.getElementById('modalLogId').value;
    const actual = document.getElementById('modalActualResult').value;
    const notes  = document.getElementById('modalNotes').value;
    if (!actual) { alert('Please enter the actual result'); return; }
    try {
        const res = await fetch(`/logs/${logId}/result`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ actual_result: parseFloat(actual), notes })
        });
        if (!res.ok) throw new Error(`server returned ${res.status}`);
        const data = await res.json();
        if (data.success) {
            closeModal();
            loadLogs(); // Refresh table
        } else {
            alert('Error: ' + data.error);
        }
    } catch (e) {
        alert('Error submitting result: ' + e.message);
    }
}

// Close modal on backdrop click or Escape key
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('resultModal')?.addEventListener('click', function(e) {
        if (e.target === this) closeModal();
    });
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
});

// ══════════════════════════════════════════════════════
// ACCURACY STATS
// ══════════════════════════════════════════════════════
async function loadAccuracy() {
    const container = document.getElementById('accuracyContainer');
    container.innerHTML = '<p class="text-gray-400 text-center py-8">Loading...</p>';
    try {
        const [accRes, retrainRes] = await Promise.all([
            fetch('/accuracy'),
            fetch('/retrain/status'),
        ]);
        if (!accRes.ok)     throw new Error(`accuracy fetch failed: ${accRes.status}`);
        if (!retrainRes.ok) throw new Error(`retrain status fetch failed: ${retrainRes.status}`);
        const accData     = await accRes.json();
        const retrainData = await retrainRes.json();
        renderAccuracy(accData, retrainData, container);
    } catch (e) {
        container.innerHTML = `<p class="text-red-500 text-center py-8">Error: ${e.message}</p>`;
    }
}

function renderAccuracy(data, retrainData, container) {
    const o = data.overall || {};
    const noData = !o.total_graded;

    if (noData) {
        container.innerHTML = `
            <div class="glass-card p-8 text-center" style="color:#64748b">
                <div class="text-5xl mb-4">📊</div>
                <p class="text-lg" style="color:#94a3b8">No graded predictions yet.</p>
                <p class="text-sm mt-1">Run analyses, then enter actual results in the Prediction Log tab.</p>
            </div>
            ${renderRetrainPanel(retrainData)}`;
        return;
    }

    const accColor = pct => pct >= 60 ? 'text-green-400' : pct >= 50 ? 'text-yellow-400' : 'text-red-400';
    const highAcc  = o.high_conf_count  ? (100 * o.high_conf_correct / o.high_conf_count).toFixed(1) : null;
    const medAcc   = o.med_conf_count   ? (100 * o.med_conf_correct  / o.med_conf_count).toFixed(1)  : null;

    container.innerHTML = `
        <!-- Overall Stats -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="glass-card p-4 text-center">
                <div class="text-xs font-semibold uppercase tracking-wider mb-1" style="color:#64748b">Overall Accuracy</div>
                <div class="text-3xl font-black ${accColor(o.accuracy_pct)}">${o.accuracy_pct ?? '—'}%</div>
                <div class="text-xs mt-1" style="color:#475569">${o.total_correct}/${o.total_graded} graded</div>
            </div>
            <div class="glass-card p-4 text-center">
                <div class="text-xs font-semibold uppercase tracking-wider mb-1" style="color:#64748b">Last 20 Picks</div>
                <div class="text-3xl font-black ${data.recent_accuracy != null ? accColor(data.recent_accuracy) : ''}" style="color:${data.recent_accuracy == null ? '#475569' : ''}">${data.recent_accuracy ?? '—'}%</div>
                <div class="text-xs mt-1" style="color:#475569">recent form</div>
            </div>
            <div class="glass-card p-4 text-center">
                <div class="text-xs font-semibold uppercase tracking-wider mb-1" style="color:#64748b">HIGH Confidence</div>
                <div class="text-3xl font-black ${highAcc != null ? accColor(highAcc) : ''}" style="color:${highAcc == null ? '#475569' : ''}">${highAcc ?? '—'}%</div>
                <div class="text-xs mt-1" style="color:#475569">${o.high_conf_count} picks</div>
            </div>
            <div class="glass-card p-4 text-center">
                <div class="text-xs font-semibold uppercase tracking-wider mb-1" style="color:#64748b">MEDIUM Confidence</div>
                <div class="text-3xl font-black ${medAcc != null ? accColor(medAcc) : ''}" style="color:${medAcc == null ? '#475569' : ''}">${medAcc ?? '—'}%</div>
                <div class="text-xs mt-1" style="color:#475569">${o.med_conf_count} picks</div>
            </div>
        </div>

        <!-- By Prop Type & By Player -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div class="glass-card p-6">
                <div class="section-header -mx-6 -mt-6 mb-4 rounded-t-2xl"><span class="section-icon">🏷️</span><h3 class="section-title">By Prop Type</h3></div>
                ${data.by_prop.length ? `
                <table class="w-full text-sm">
                    <thead>
                        <tr class="games-thead">
                            <th class="games-th text-left">Prop</th>
                            <th class="games-th text-right">Picks</th>
                            <th class="games-th text-right">Accuracy</th>
                            <th class="games-th text-right">Avg Edge</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.by_prop.map(r => `
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
                                <td class="py-2" style="color:#cbd5e1">${getPropTypeLabel(r.prop_type)}</td>
                                <td class="py-2 text-right" style="color:#475569">${r.total}</td>
                                <td class="py-2 text-right font-bold ${accColor(r.accuracy_pct)}">${r.accuracy_pct}%</td>
                                <td class="py-2 text-right ${r.avg_edge_pct > 0 ? 'text-green-400' : 'text-red-400'}">${r.avg_edge_pct > 0 ? '+' : ''}${r.avg_edge_pct}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>` : '<p class="text-sm" style="color:#475569">No data yet</p>'}
            </div>

            <div class="glass-card p-6">
                <div class="section-header -mx-6 -mt-6 mb-4 rounded-t-2xl"><span class="section-icon">🌟</span><h3 class="section-title">Top Players Analyzed</h3></div>
                ${data.by_player.length ? `
                <table class="w-full text-sm">
                    <thead>
                        <tr class="games-thead">
                            <th class="games-th text-left">Player</th>
                            <th class="games-th text-right">Picks</th>
                            <th class="games-th text-right">Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.by_player.map(r => `
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
                                <td class="py-2" style="color:#cbd5e1">${r.player_name || r.player_id}</td>
                                <td class="py-2 text-right" style="color:#475569">${r.total}</td>
                                <td class="py-2 text-right font-bold ${accColor(r.accuracy_pct)}">${r.accuracy_pct}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>` : '<p class="text-sm" style="color:#475569">No data yet</p>'}
            </div>
        </div>
        <p class="text-xs text-center mb-6" style="color:#475569">Total predictions logged: ${data.total_predictions} (${o.total_graded} graded)</p>

        ${renderRetrainPanel(retrainData)}
    `;
}

// ── Retrain metadata panel ────────────────────────────────────────────────────
function renderRetrainPanel(retrainData) {
    const meta    = retrainData?.meta    || {};
    const running = retrainData?.running || false;
    const last    = retrainData?.last_result;

    const lastRetrainAt = meta.last_retrain_at
        ? new Date(meta.last_retrain_at).toLocaleString()
        : 'Never';
    const samplesAtLast = meta.samples_at_last_retrain ?? 0;
    const lastAuc  = meta.last_auc  != null ? meta.last_auc.toFixed(3)  : '—';
    const lastRmse = meta.last_rmse != null ? meta.last_rmse.toFixed(3) : '—';

    let lastResultHtml = '';
    if (running) {
        lastResultHtml = `<span class="text-blue-500 font-medium animate-pulse">⏳ Retraining in progress…</span>`;
    } else if (last) {
        if (last.status === 'retrained') {
            lastResultHtml = `<span class="text-green-600 font-medium">✓ Retrained — ${last.total_samples} total samples (${last.log_samples} from logs × 2)</span>`;
        } else if (last.status === 'skipped') {
            lastResultHtml = `<span class="text-yellow-600">⚠ Skipped — ${last.reason}</span>`;
        } else if (last.status === 'error') {
            lastResultHtml = `<span class="text-red-600">✗ Error — ${last.error}</span>`;
        }
    }

    return `
        <div class="glass-card mt-6">
            <div class="section-header">
                <span class="section-icon">🤖</span>
                <h3 class="section-title">Model Retraining</h3>
                <button onclick="triggerRetrain()"
                        id="retrainBtn"
                        class="pill-btn purple ml-auto ${running ? 'opacity-50 cursor-not-allowed' : ''}">
                    ${running ? '⏳ Running…' : '🔄 Retrain Now'}
                </button>
            </div>
            <div class="p-6">
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div class="inner-card text-center">
                        <div class="inner-card-title">Last Retrain</div>
                        <div class="text-sm font-semibold" style="color:#cbd5e1">${lastRetrainAt}</div>
                    </div>
                    <div class="inner-card text-center">
                        <div class="inner-card-title">Samples Used</div>
                        <div class="text-sm font-semibold" style="color:#cbd5e1">${samplesAtLast > 0 ? samplesAtLast.toLocaleString() : '—'}</div>
                    </div>
                    <div class="inner-card text-center">
                        <div class="inner-card-title">Model AUC</div>
                        <div class="text-sm font-bold ${lastAuc !== '—' ? (parseFloat(lastAuc) >= 0.6 ? 'text-green-400' : 'text-yellow-400') : ''}" style="color:${lastAuc === '—' ? '#475569' : ''}">${lastAuc}</div>
                    </div>
                    <div class="inner-card text-center">
                        <div class="inner-card-title">Model RMSE</div>
                        <div class="text-sm font-semibold" style="color:#cbd5e1">${lastRmse}</div>
                    </div>
                </div>
                ${lastResultHtml ? `<div class="text-sm mt-2">${lastResultHtml}</div>` : ''}
                <p class="text-xs mt-3" style="color:#475569">
                    Retrains automatically every night at ~4 AM if ≥ 50 new graded predictions exist.
                    Manual retrain triggers immediately (threshold: 10 new samples).
                </p>
            </div>
        </div>
    `;
}

async function triggerRetrain() {
    const btn = document.getElementById('retrainBtn');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Starting…'; }
    try {
        const res = await fetch('/retrain', { method: 'POST' });
        if (res.status === 409) {
            alert('Retrain is already running. Check back shortly.');
            if (btn) { btn.disabled = false; btn.textContent = '🔄 Retrain Now'; }
            return;
        }
        if (!res.ok) throw new Error(`retrain start failed: ${res.status}`);
        // poll every few seconds; give up after 30 failures (~2 min)
        let pollFailures = 0;
        const poll = setInterval(async () => {
            try {
                const sr = await fetch('/retrain/status');
                if (!sr.ok) throw new Error(`status ${sr.status}`);
                const sd = await sr.json();
                pollFailures = 0;
                if (!sd.running) {
                    clearInterval(poll);
                    if (btn) { btn.disabled = false; btn.textContent = '🔄 Retrain Now'; }
                    loadAccuracy();
                }
            } catch (pollErr) {
                console.warn('Retrain status poll failed:', pollErr);
                pollFailures++;
                if (pollFailures >= 30) {
                    clearInterval(poll);
                    if (btn) { btn.disabled = false; btn.textContent = '🔄 Retrain Now'; }
                    console.error('Gave up polling retrain status after 30 failures.');
                }
            }
        }, 4000);
    } catch (e) {
        alert('Error starting retrain: ' + e.message);
        if (btn) { btn.disabled = false; btn.textContent = '🔄 Retrain Now'; }
    }
}

async function loadBias() {
    const container = document.getElementById('biasContainer');
    container.innerHTML = '<p class="text-gray-400 text-center py-8">Loading...</p>';
    try {
        const res = await fetch('/bias');
        if (!res.ok) throw new Error(`bias fetch failed: ${res.status}`);
        const data = await res.json();

        if (!data || (!data.by_prop && !data.by_location && !data.by_confidence)) {
            container.innerHTML = '<p class="text-gray-400 text-center py-8">Not enough graded predictions yet (need at least 10 per prop type).</p>';
            return;
        }

        const pct = v => v != null ? `${v.toFixed(1)}%` : '—';
        const num = v => v != null ? v.toFixed(2) : '—';
        const errorBadge = err => {
            if (err == null) return '—';
            const cls = Math.abs(err) < 0.5 ? 'text-green-600' : Math.abs(err) < 1.5 ? 'text-yellow-600' : 'text-red-600';
            return `<span class="${cls} font-medium">${err > 0 ? '+' : ''}${err.toFixed(2)}</span>`;
        };
        const calBadge = (avgProb, actualRate) => {
            if (avgProb == null || actualRate == null) return '—';
            const diff = (avgProb - actualRate) * 100;
            const cls = Math.abs(diff) < 5 ? 'text-green-600' : Math.abs(diff) < 12 ? 'text-yellow-600' : 'text-red-600';
            return `<span class="${cls} text-xs">${diff > 0 ? '+' : ''}${diff.toFixed(1)}% (pred vs actual over rate)</span>`;
        };

        let html = '';

        // by prop
        if (data.by_prop && data.by_prop.length) {
            html += `
            <div class="glass-card mb-6">
                <div class="card-header-bar">
                    <span class="card-header-dot orange"></span><span class="card-header-dot yellow"></span><span class="card-header-dot green"></span>
                    <span class="card-header-label">Bias by Prop Type</span>
                </div>
                <div class="p-4 overflow-x-auto">
                    <p class="text-xs text-gray-400 mb-3">Avg Error = predicted − actual. Positive = over-predicting. Calibration = how far avg probability is from actual over rate.</p>
                    <table class="min-w-full text-sm">
                        <thead class="bg-gray-50 text-xs uppercase text-gray-500">
                            <tr>
                                <th class="px-3 py-2 text-left">Prop</th>
                                <th class="px-3 py-2 text-right">n</th>
                                <th class="px-3 py-2 text-right">Accuracy</th>
                                <th class="px-3 py-2 text-right">Avg Error</th>
                                <th class="px-3 py-2 text-right">MAE</th>
                                <th class="px-3 py-2 text-left">Calibration</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.by_prop.map(r => `
                            <tr class="border-t border-gray-100 hover:bg-gray-50">
                                <td class="px-3 py-2 font-medium">${getPropTypeLabel(r.prop_type)}</td>
                                <td class="px-3 py-2 text-right text-gray-500">${r.n}</td>
                                <td class="px-3 py-2 text-right">${pct(r.accuracy_pct)}</td>
                                <td class="px-3 py-2 text-right">${errorBadge(r.avg_error)}</td>
                                <td class="px-3 py-2 text-right text-gray-500">${num(r.mae)}</td>
                                <td class="px-3 py-2">${calBadge(r.avg_prob, r.actual_over_rate)}</td>
                            </tr>`).join('')}
                        </tbody>
                    </table>
                </div>
            </div>`;
        }

        // by location
        if (data.by_location && data.by_location.length) {
            html += `
            <div class="glass-card mb-6">
                <div class="card-header-bar">
                    <span class="card-header-dot orange"></span><span class="card-header-dot yellow"></span><span class="card-header-dot green"></span>
                    <span class="card-header-label">Bias by Home / Away</span>
                </div>
                <div class="p-4 overflow-x-auto">
                    <table class="min-w-full text-sm">
                        <thead class="bg-gray-50 text-xs uppercase text-gray-500">
                            <tr>
                                <th class="px-3 py-2 text-left">Location</th>
                                <th class="px-3 py-2 text-right">n</th>
                                <th class="px-3 py-2 text-right">Accuracy</th>
                                <th class="px-3 py-2 text-right">Avg Error</th>
                                <th class="px-3 py-2 text-left">Calibration</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.by_location.map(r => `
                            <tr class="border-t border-gray-100 hover:bg-gray-50">
                                <td class="px-3 py-2 font-medium capitalize">${r.location}</td>
                                <td class="px-3 py-2 text-right text-gray-500">${r.n}</td>
                                <td class="px-3 py-2 text-right">${pct(r.accuracy_pct)}</td>
                                <td class="px-3 py-2 text-right">${errorBadge(r.avg_error)}</td>
                                <td class="px-3 py-2">${calBadge(r.avg_prob, r.actual_over_rate)}</td>
                            </tr>`).join('')}
                        </tbody>
                    </table>
                </div>
            </div>`;
        }

        // by confidence tier
        if (data.by_confidence && data.by_confidence.length) {
            html += `
            <div class="glass-card mb-6">
                <div class="card-header-bar">
                    <span class="card-header-dot orange"></span><span class="card-header-dot yellow"></span><span class="card-header-dot green"></span>
                    <span class="card-header-label">Confidence Tier Calibration</span>
                </div>
                <div class="p-4 overflow-x-auto">
                    <p class="text-xs text-gray-400 mb-3">HIGH should hit >60%, MEDIUM >52%. If not, the model will auto-adjust thresholds on next retrain.</p>
                    <table class="min-w-full text-sm">
                        <thead class="bg-gray-50 text-xs uppercase text-gray-500">
                            <tr>
                                <th class="px-3 py-2 text-left">Confidence</th>
                                <th class="px-3 py-2 text-right">n</th>
                                <th class="px-3 py-2 text-right">Accuracy</th>
                                <th class="px-3 py-2 text-left">Calibration</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.by_confidence.map(r => {
                                const target = r.confidence === 'HIGH' ? 60 : r.confidence === 'MEDIUM' ? 52 : 0;
                                const ok = r.accuracy_pct != null && r.accuracy_pct >= target;
                                const cls = ok ? 'text-green-600' : 'text-red-600';
                                return `
                                <tr class="border-t border-gray-100 hover:bg-gray-50">
                                    <td class="px-3 py-2 font-medium ${cls}">${r.confidence}</td>
                                    <td class="px-3 py-2 text-right text-gray-500">${r.n}</td>
                                    <td class="px-3 py-2 text-right ${cls} font-bold">${pct(r.accuracy_pct)}</td>
                                    <td class="px-3 py-2">${calBadge(r.avg_prob, r.actual_over_rate)}</td>
                                </tr>`;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>`;
        }

        container.innerHTML = html || '<p class="text-gray-400 text-center py-8">No bias data yet.</p>';
    } catch (e) {
        container.innerHTML = `<p class="text-red-400 text-center py-8">Error loading bias report: ${e.message}</p>`;
    }
}