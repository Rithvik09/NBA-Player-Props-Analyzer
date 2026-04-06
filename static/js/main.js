let selectedPlayerId = null;
let performanceChart = null;
let toastTimeout = null;
let selectedLocation = 'auto';

async function autoFillOpponent(playerId) {
    const opponentSelect = document.getElementById('opponentTeam');
    const locationNote   = document.getElementById('locationNote');

    if (locationNote) {
        locationNote.textContent = "Detecting today's game...";
        locationNote.className = 'text-xs text-gray-400 mt-1';
    }

    try {
        const res  = await fetch(`/player_game_info/${playerId}`);
        const data = await res.json();

        if (data.opponent_team_id) {
            // Auto-select the opponent in the dropdown
            opponentSelect.value = data.opponent_team_id;

            // Auto-set home/away
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
    // isHome: true = Home, false = Away, null = Auto
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
    const locationBtns = document.querySelectorAll('.location-btn');
    
    // Location button handlers
    locationBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            locationBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            selectedLocation = this.dataset.location;
        });
    });
    
    // Player search functionality
    let searchTimeout = null;
    playerSearch.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        selectedPlayerId = null;
        
        const query = this.value.trim();
        
        if (query.length < 2) {
            suggestions.innerHTML = '<div style="padding: 1rem; color: #94a3b8;">Type at least 2 characters to search...</div>';
            suggestions.style.display = 'block';
            return;
        }
        
        suggestions.innerHTML = '<div style="padding: 1rem; color: #94a3b8;">Searching...</div>';
        suggestions.style.display = 'block';
        
        searchTimeout = setTimeout(() => {
            fetch(`/search_players?q=${encodeURIComponent(query)}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Search failed');
                    }
                    return response.json();
                })
                .then(players => {
                    suggestions.innerHTML = '';
                    if (!players || players.length === 0) {
                        suggestions.innerHTML = '<div style="padding: 1rem; color: #94a3b8;">No players found</div>';
                    } else {
                        players.forEach(player => {
                            const div = document.createElement('div');
                            div.style.cssText = 'padding: 0.75rem 1rem; cursor: pointer; border-bottom: 1px solid #f3f4f6; transition: background-color 0.15s;';
                            div.textContent = player.full_name;
                            div.addEventListener('mouseenter', function() {
                                this.style.backgroundColor = '#f9fafb';
                            });
                            div.addEventListener('mouseleave', function() {
                                this.style.backgroundColor = 'transparent';
                            });
                            div.addEventListener('click', function(e) {
                                e.stopPropagation();
                                playerSearch.value = player.full_name;
                                selectedPlayerId = player.id;
                                suggestions.classList.add('hidden');
                                autoFillOpponent(player.id);
                            });
                            suggestions.appendChild(div);
                        });
                        // Remove border from last item
                        if (suggestions.lastElementChild) {
                            suggestions.lastElementChild.style.borderBottom = 'none';
                        }
                    }
                })
                .catch(error => {
                    console.error('Error searching players:', error);
                    suggestions.innerHTML = '<div style="padding: 1rem; color: #ef4444;">Error loading players. Please try again.</div>';
                });
        }, 300);
    });

    // Close suggestions on click outside
    document.addEventListener('click', function(e) {
        if (!suggestions.contains(e.target) && e.target !== playerSearch && !playerSearch.contains(e.target)) {
            suggestions.style.display = 'none';
        }
    });
    
    // Keep suggestions open when clicking inside
    suggestions.addEventListener('click', function(e) {
        e.stopPropagation();
    });

    // Analyze prop button handler
    analyzePropBtn.addEventListener('click', async function() {
        if (!selectedPlayerId) {
            showToast('Player Required', 'Please select a player from the search results.', 'error');
            return;
        }
        
        const propType = document.getElementById('propType').value;
        const line = document.getElementById('lineInput').value;
        const opponentTeamId = document.getElementById('opponentTeam').value;
        
        if (!line) {
            showToast('Line Required', 'Please enter a betting line (e.g., 25.5).', 'error');
            return;
        }
        
        if (!opponentTeamId) {
            showToast('Opponent Required', 'Please select an opponent team.', 'error');
            return;
        }
        
        try {
            analyzePropBtn.disabled = true;
            analyzePropBtn.innerHTML = '<span class="loading"></span> <span>Analyzing...</span>';

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

            const stats = analysis.player_stats;
            if (!stats) {
                throw new Error('Missing player stats');
            }
            
            updateResults(analysis, stats, propType, parseFloat(line));
            
        } catch (error) {
            console.error('Error:', error);
            showToast('Analysis Failed', error.message || 'Please try again.', 'error');
        } finally {
            analyzePropBtn.disabled = false;
            analyzePropBtn.innerHTML = '<span class="analyze-btn-inner">⚡ Analyze Prop</span>';
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

        // Update the location note to show what was detected
        const locationNote = document.getElementById('locationNote');
        if (locationNote && analysis.location_detected) {
            locationNote.textContent = analysis.is_home
                ? '🏠 Auto-detected: Home game'
                : '✈️ Auto-detected: Away game';
            locationNote.className = 'text-xs text-blue-500 mt-1 font-medium';
        }
        
    // Update main recommendation
    const mainRec = document.getElementById('mainRecommendation');
    mainRec.textContent = analysis.recommendation;
    
    // Update badges
    const modelBadge = document.getElementById('modelBadge');
    
    // Extract model source - it can be a string or an object with 'source' field
    let modelSource = 'heuristic';
    if (analysis.model_used) {
        if (typeof analysis.model_used === 'string') {
            modelSource = analysis.model_used;
        } else if (typeof analysis.model_used === 'object' && analysis.model_used.source) {
            modelSource = analysis.model_used.source;
        }

        updateMLAnalysis(analysis, stats, propType);
        updatePlayerContext(analysis.context?.player, stats, propType, analysis);
        updateTeamContext(analysis.context?.team);
    updateMatchupAnalysis(analysis.context?.player?.matchup_history, analysis.context?.player?.position_matchup);
    
    // Update chart and table
        updatePerformanceChart(stats, propType, line);
        updateRecentGames(stats, propType, line);
}


    const fb = analysis.factor_breakdown || {};
    
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

    // Gather stat data for this prop — combined props live under stats.combined_stats
    const propStats = propType && propType in stats
        ? (stats[propType] || {})
        : (propType && stats.combined_stats && propType in stats.combined_stats)
            ? (stats.combined_stats[propType] || {})
            : (stats['points'] || {});
    const seasonAvg  = propStats.avg       ?? null;
    const last5Avg   = propStats.last5_avg ?? null;
    const trend      = analysis.trend      || {};

    // ── Factor scoring ──────────────────────────────────────────────────────
    // Each factor produces a { label, summary, strength, bullish } object.
    // strength: 'strong' | 'moderate' | 'weak'
    // bullish: true = favours OVER, false = favours UNDER
    const factors = [];

    // 0. Home/Away split
    const isHome = analysis.is_home ?? null;  // null = unknown, don't guess
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

    // 1. Recent trend
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

    // 2. Hot/cold streak (last 5 vs season avg)
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

    // 3. Historical hit rate
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

    // 4. Matchup history vs this opponent
    if (matchup && matchup.games_played > 0) {
        // Compute the right combined average depending on prop type
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
        // Higher defensive rating = worse defense = bullish for player
        const isBullish = defRtg > 110;
        factors.push({
            label: 'Opponent Positional Defense',
            summary: ptsAllowed != null
                ? `Opponent allows ${ptsAllowed.toFixed(1)} pts/game to this position (def rating: ${defRtg.toFixed(0)})`
                : `Opponent defensive rating vs position: ${defRtg.toFixed(0)}`,
            strength: defRtg > 115 || defRtg < 105 ? 'strong' : 'moderate',
            bullish: isBullish
        });
    }

    // 6. Opponent injury impact
    if (opponent.injury_impact != null && opponent.injury_impact > 0.05) {
        const impact = opponent.injury_impact;
        factors.push({
            label: 'Opponent Injuries',
            summary: `Opponent missing key personnel (injury impact: ${(impact * 100).toFixed(0)}%) — weakened defense`,
            strength: impact > 0.3 ? 'strong' : impact > 0.15 ? 'moderate' : 'weak',
            bullish: true
        });
    }

    // 7. Team injury impact (hurts the player)
    if (team.injury_impact != null && team.injury_impact > 0.05) {
        const impact = team.injury_impact;
        factors.push({
            label: 'Team Injuries',
            summary: `Player\'s own team is short-handed (injury impact: ${(impact * 100).toFixed(0)}%) — may affect usage/pace`,
            strength: impact > 0.3 ? 'strong' : impact > 0.15 ? 'moderate' : 'weak',
            bullish: false
        });
    }

    // 8. Rest days
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

    // ── Sort: strongest factors first ────────────────────────────────────────
    const strengthOrder = { strong: 0, moderate: 1, weak: 2 };
    factors.sort((a, b) => strengthOrder[a.strength] - strengthOrder[b.strength]);

    // ── Build narrative ───────────────────────────────────────────────────────
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

    // ── Build injury listing helpers ─────────────────────────────────────────
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

    // Prediction Model / Value Analysis cards
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
            { label: 'FG%', value: shooting.fg_pct_recent ? `${(shooting.fg_pct_recent * 100).toFixed(1)}%` : 'N/A' },
            { label: 'Win Rate (L10)', value: stats.impact?.win_rate_last10 ? `${(stats.impact.win_rate_last10 * 100).toFixed(0)}%` : 'N/A' },
        ];
        
        if (schedule.is_back_to_back) {
            items.push({ label: '⚠️ Schedule', value: 'Back-to-back game', class: 'text-red-600 font-bold' });
        }
        
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'context-item' + (item.class ? ' ' + item.class : '');
            div.innerHTML = `
                <span class="context-label">${item.label}</span>
                <span class="context-value">${item.value}</span>
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
            { label: 'Defensive Rating', value: teamContext.defensive_rating?.toFixed(1) || 'N/A' },
            { label: 'Injury Impact', value: `${(teamContext.injury_impact * 100).toFixed(1)}%`,
              class: teamContext.injury_impact > 0.15 ? 'text-red-600 font-bold' : '' }
        ];
        
        if (teamContext.injuries && teamContext.injuries.total_players_out > 0) {
            items.push({
                label: 'Players Out',
                value: `${teamContext.injuries.key_players_out} key`,
                class: 'text-red-600'
            });
        }
        
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'context-item' + (item.class ? ' ' + item.class : '');
            div.innerHTML = `
                <span class="context-label">${item.label}</span>
                <span class="context-value">${item.value}</span>
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
        
            items.forEach(item => {
                const div = document.createElement('div');
            div.className = 'context-item';
                div.innerHTML = `
                <span class="context-label">${item.label}</span>
                <span class="context-value">${item.value}</span>
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
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }) || [];
    
    const reversedValues = [...values].reverse();
    const reversedDates = [...dates].reverse();
    
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: reversedDates,
            datasets: [
                {
                    label: 'Actual Performance',
                    data: reversedValues,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                },
                {
                    label: 'Betting Line',
                    data: Array(reversedDates.length).fill(line),
                    borderColor: '#ef4444',
                    borderDash: [8, 4],
                    tension: 0,
                    fill: false,
                    borderWidth: 2,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 13,
                            weight: 600,
                            family: 'Inter'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 12,
                            family: 'Inter'
                        }
                    }
                },
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        font: {
                            size: 11,
                            family: 'Inter'
                        }
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

function showToast(title, message, variant = 'success') {
    const toast = document.getElementById('toast');
    const toastTitle = document.getElementById('toastTitle');
    const toastMessage = document.getElementById('toastMessage');
    
    if (!toast || !toastTitle || !toastMessage) return;

    toast.classList.remove('hidden', 'toast-success', 'toast-error');
    toast.classList.add(variant === 'error' ? 'toast-error' : 'toast-success');
    toastTitle.textContent = title || '';
    toastMessage.textContent = message || '';

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.add('hidden');
    }, 4000);
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
    ['analyzer', 'logs', 'accuracy'].forEach(t => {
        document.getElementById(`tab-content-${t}`).classList.toggle('hidden', t !== tab);
        const btn = document.getElementById(`tab-${t}`);
        if (btn) {
            btn.className = t === tab ? 'tab-btn tab-active' : 'tab-btn';
        }
    });
    if (tab === 'logs') loadLogs();
    if (tab === 'accuracy') loadAccuracy();
}

// ══════════════════════════════════════════════════════
// PREDICTION LOGS
// ══════════════════════════════════════════════════════
async function triggerAutoGrade() {
    const status = document.getElementById('autoGradeStatus');
    if (status) status.textContent = 'Fetching results...';
    try {
        await fetch('/logs/auto-grade', { method: 'POST' });
        // Poll for completion — results come back via background thread
        // Just wait a few seconds then refresh
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
        // correct is null — either PASS (result already filled) or ungraded
        if (row.actual_result != null) {
            // PASS prediction with a known result — show badge only
            return `<span class="text-xs bg-gray-100 text-gray-500 px-2 py-1 rounded">PASS</span>`;
        }
        // No result yet — allow manual entry
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
        // Poll until done (give up after 30 consecutive failures ~2 minutes)
        let pollFailures = 0;
        const poll = setInterval(async () => {
            try {
                const sr = await fetch('/retrain/status');
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