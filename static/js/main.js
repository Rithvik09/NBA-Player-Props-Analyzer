let selectedPlayerId = null;
let performanceChart = null;
let toastTimeout = null;
let selectedLocation = 'auto';

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
                                suggestions.style.display = 'none';
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
                    game_location: selectedLocation
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
            
            updateResults(analysis, stats, propType, line);
            showToast('Analysis Complete', 'Scroll down to view results.', 'success');
            
            // Smooth scroll to results
            setTimeout(() => {
                document.getElementById('results').scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 300);
            
        } catch (error) {
            console.error('Error:', error);
            showToast('Analysis Failed', error.message || 'Please try again.', 'error');
        } finally {
            analyzePropBtn.disabled = false;
            analyzePropBtn.innerHTML = `
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                </svg>
                <span>Analyze Prop Bet</span>
            `;
        }
    });
});

function updateResults(analysis, stats, propType, line) {
        const resultsSection = document.getElementById('results');
        resultsSection.classList.remove('hidden');
        
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
    }
    
    const modelLabels = {
        'trained_classifier_calibrated': '🤖 ML Model (5-Season Trained)',
        'trained_rf_classifier': '🤖 ML Model (5-Season Trained)',
        'incremental_sgd': '⚡ ML Model (Incremental)',
        'heuristic': '📊 Heuristic Model'
    };
    modelBadge.textContent = modelLabels[modelSource] || `📊 ${modelSource}`;
    modelBadge.title = (modelSource.includes('trained') || modelSource.includes('classifier')) ? 
        '186 features trained on 5 seasons of NBA data' :
        modelSource === 'incremental_sgd' ?
        'Continuously learning model updated with recent data' :
        'Rule-based model with statistical adjustments';
    
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.textContent = `${analysis.confidence} Confidence`;
    confidenceBadge.className = 'badge badge-' + 
        (analysis.confidence === 'HIGH' ? 'success' : 
         analysis.confidence === 'MEDIUM' ? 'warning' : 'danger');
    
    // Update model info
    const modelInfo = document.getElementById('modelInfo');
    let modelTimeStr = 'N/A';
    let dataTimeStr = 'N/A';
    
    // Extract model update time from model_used object if present
    try {
        if (analysis.model_used && typeof analysis.model_used === 'object') {
            if (analysis.model_used.updated_at) {
                modelTimeStr = new Date(analysis.model_used.updated_at * 1000).toLocaleDateString();
            } else if (analysis.model_used.updated_at_iso) {
                modelTimeStr = new Date(analysis.model_used.updated_at_iso).toLocaleDateString();
            }
        } else if (analysis.model_last_updated) {
            if (typeof analysis.model_last_updated === 'number') {
                modelTimeStr = new Date(analysis.model_last_updated * 1000).toLocaleDateString();
            } else if (typeof analysis.model_last_updated === 'string') {
                modelTimeStr = new Date(analysis.model_last_updated).toLocaleDateString();
            }
        }
    } catch (e) {
        console.error('Error parsing model update time:', e);
        modelTimeStr = 'Recent';
    }
    
    try {
        if (analysis.precomputed_last_updated) {
            if (typeof analysis.precomputed_last_updated === 'number') {
                dataTimeStr = new Date(analysis.precomputed_last_updated * 1000).toLocaleDateString();
            } else if (typeof analysis.precomputed_last_updated === 'string') {
                dataTimeStr = new Date(analysis.precomputed_last_updated).toLocaleDateString();
            }
        }
    } catch (e) {
        console.error('Error parsing precomputed_last_updated:', e);
        dataTimeStr = 'Recent';
    }
    
    const modelTypeDesc = (modelSource.includes('trained') || modelSource.includes('classifier')) ? 
        '5-Season Trained Model with 186 Features' : 
        modelSource === 'incremental_sgd' ? 
        'Incremental Learning Model' : 
        'Advanced Heuristic Engine';
    
    modelInfo.innerHTML = `
        <div style="font-size: 0.95rem; line-height: 1.6;">
            <strong>🤖 Model:</strong> ${modelTypeDesc}<br>
            <strong>📅 Trained:</strong> ${modelTimeStr} • <strong>📊 Data Updated:</strong> ${dataTimeStr}
        </div>
    `;
    
    // Update metrics
    document.getElementById('predictedValue').textContent = analysis.predicted_value.toFixed(1);
    document.getElementById('edgeValue').textContent = 
        `${analysis.edge > 0 ? '+' : ''}${(analysis.edge * 100).toFixed(1)}% edge`;
    document.getElementById('edgeValue').className = 
        `metric-change ${analysis.edge > 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('overProbability').textContent = 
        `${(analysis.over_probability * 100).toFixed(1)}%`;
    
    document.getElementById('hitRate').textContent = 
        `${(analysis.hit_rate * 100).toFixed(1)}%`;
    document.getElementById('hitRateDetails').textContent = 
        `${analysis.times_hit} of ${analysis.total_games} games`;
    
    // Update explanation
    updateExplanation(analysis, propType, line);
    
    // Update context sections
    updatePlayerContext(analysis.context?.player, stats, analysis.factor_breakdown);
        updateTeamContext(analysis.context?.team);
    updateMatchupAnalysis(analysis.context?.player?.matchup_history, analysis.context?.player?.position_matchup);
    
    // Update chart and table
        updatePerformanceChart(stats, propType, line);
        updateRecentGames(stats, propType, line);
}

function updateExplanation(analysis, propType, line) {
    const el = document.getElementById('resultExplanation');
    if (!el) return;

    const fb = analysis.factor_breakdown || {};
    
    // Extract model source properly
    let modelSource = 'heuristic';
    if (analysis.model_used) {
        if (typeof analysis.model_used === 'string') {
            modelSource = analysis.model_used;
        } else if (typeof analysis.model_used === 'object' && analysis.model_used.source) {
            modelSource = analysis.model_used.source;
        }
    }
    
    const propLabel = getPropTypeLabel(propType);
    const prob = (analysis.over_probability * 100).toFixed(1);
    const pred = analysis.predicted_value.toFixed(1);
    const edge = (analysis.edge * 100).toFixed(1);
    
    // Get factor analysis early (before it's used)
    const factorAnalysis = analysis.factor_analysis || {};
    
    // Build comprehensive explanation sections
    let explanationHTML = '<div style="space-y: 1.5rem;">';
    
    // 1. Main Prediction Summary
    const modelDesc = (modelSource.includes('trained') || modelSource.includes('classifier')) ? 
        'trained machine learning model (186 features, 5 seasons of data)' : 
        modelSource === 'incremental_sgd' ? 
        'incrementally-trained machine learning model' : 
        'advanced analytics engine';
    
    // Check for injury alerts first
    const teamContext = analysis.context?.team;
    const ownInjuries = teamContext?.injuries;
    const oppInjuries = teamContext?.opp_injuries;
    
    // Show injury alert box if there are significant injuries
    if ((ownInjuries && ownInjuries.key_players_out > 0) || (oppInjuries && oppInjuries.key_players_out > 0)) {
        explanationHTML += `
            <div style="margin-bottom: 1.5rem; padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; border-radius: 0.5rem;">
                <strong style="font-size: 1.1rem;">🏥 Injury Report</strong><br>
                <div style="margin-top: 0.5rem;">
        `;
        
        if (ownInjuries && ownInjuries.total_players_out > 0) {
            const activeInjuries = ownInjuries.active_injuries || [];
            if (activeInjuries.length > 0) {
                const playerNames = activeInjuries.map(inj => {
                    const name = inj.player_name || inj.name || 'Unknown';
                    const status = inj.status || 'Out';
                    return `${name} (${status})`;
                });
                explanationHTML += `<p><strong>Team:</strong> ${playerNames.join(', ')}</p>`;
            } else {
                explanationHTML += `<p><strong>Team:</strong> ${ownInjuries.total_players_out} player${ownInjuries.total_players_out === 1 ? '' : 's'} out</p>`;
            }
        }
        
        if (oppInjuries && oppInjuries.total_players_out > 0) {
            const activeInjuries = oppInjuries.active_injuries || [];
            if (activeInjuries.length > 0) {
                const playerNames = activeInjuries.map(inj => {
                    const name = inj.player_name || inj.name || 'Unknown';
                    const status = inj.status || 'Out';
                    return `${name} (${status})`;
                });
                explanationHTML += `<p><strong>Opponent:</strong> ${playerNames.join(', ')}</p>`;
            } else {
                explanationHTML += `<p><strong>Opponent:</strong> ${oppInjuries.total_players_out} player${oppInjuries.total_players_out === 1 ? '' : 's'} out</p>`;
            }
        }
        
        explanationHTML += `
                </div>
            </div>
        `;
    }
    
    explanationHTML += `
        <div style="margin-bottom: 1.5rem; padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-left: 4px solid #3b82f6; border-radius: 0.5rem;">
            <strong style="font-size: 1.1rem;">📊 Prediction Summary</strong><br>
            <p style="margin-top: 0.5rem;">Our ${modelDesc} predicts <strong style="color: #3b82f6;">${pred} ${propLabel}</strong> against a line of <strong>${parseFloat(line).toFixed(1)}</strong>. 
            This gives a <strong style="color: ${prob >= 60 ? '#10b981' : prob <= 40 ? '#ef4444' : '#f59e0b'};">${prob}% probability</strong> to go over, 
            representing a <strong style="color: ${edge > 0 ? '#10b981' : '#ef4444'};">${edge > 0 ? '+' : ''}${edge}% edge</strong>.</p>
            ${factorAnalysis.detailed_explanation ? `<p style="margin-top: 0.5rem; font-style: italic; color: #64748b;">${factorAnalysis.detailed_explanation}</p>` : ''}
        </div>
    `;
    
    // 2. Comprehensive Factor Analysis (NEW - uses all 186 features)
    const posDrivers = [];
    const negDrivers = [];
    let keyDriversSection = '';
    
    // Use comprehensive factor analysis if available
    if (factorAnalysis.positive_factors && factorAnalysis.positive_factors.length > 0) {
        factorAnalysis.positive_factors.forEach(factor => {
            const impactStr = factor.impact > 0 ? `+${factor.impact.toFixed(1)}` : `${factor.impact.toFixed(1)}`;
            posDrivers.push(`✅ <strong>${factor.name}</strong> (${factor.value}): ${factor.explanation} [${impactStr} pts]`);
        });
    }
    
    if (factorAnalysis.negative_factors && factorAnalysis.negative_factors.length > 0) {
        factorAnalysis.negative_factors.forEach(factor => {
            const impactStr = factor.impact < 0 ? `${factor.impact.toFixed(1)}` : `-${factor.impact.toFixed(1)}`;
            negDrivers.push(`⚠️ <strong>${factor.name}</strong> (${factor.value}): ${factor.explanation} [${impactStr} pts]`);
        });
    }
    
    // Add Key Drivers section if available
    if (factorAnalysis.key_drivers && factorAnalysis.key_drivers.length > 0) {
        const topDrivers = factorAnalysis.key_drivers.slice(0, 5);
        keyDriversSection = `
            <div style="margin-bottom: 1.5rem; padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; border-radius: 0.5rem;">
                <strong style="font-size: 1.1rem;">🎯 Top 5 Key Drivers</strong><br>
                <ul style="margin-top: 0.5rem; padding-left: 1.5rem; line-height: 1.8;">
                    ${topDrivers.map(d => {
                        const impactStr = d.impact > 0 ? `+${d.impact.toFixed(1)}` : `${d.impact.toFixed(1)}`;
                        const icon = d.impact > 0 ? '✅' : '⚠️';
                        return `<li>${icon} <strong>${d.name}</strong> (${d.value}): ${d.explanation} [${impactStr} pts]</li>`;
                    }).join('')}
                </ul>
            </div>
        `;
    }
    
    // Add Key Drivers section before positive/negative factors
    if (keyDriversSection) {
        explanationHTML += keyDriversSection;
    }
    
    // Fallback to manual extraction if factor_analysis not available
    if (posDrivers.length === 0 && negDrivers.length === 0) {
        if (fb.trend_adj && Math.abs(fb.trend_adj) > 0.15) {
            const impact = fb.trend_adj.toFixed(2);
            if (fb.trend_adj > 0) {
                posDrivers.push(`📈 <strong>Recent Form</strong>: +${impact} pts (trending up)`);
            } else {
                negDrivers.push(`📉 <strong>Recent Form</strong>: ${impact} pts (trending down)`);
            }
        }
        
        if (fb.fatigue_adj && Math.abs(fb.fatigue_adj) > 0.1) {
        const impact = fb.fatigue_adj.toFixed(2);
        if (fb.is_back_to_back) {
            negDrivers.push(`😴 <strong>Back-to-Back Game</strong>: ${impact} pts (fatigue factor)`);
        } else if (fb.games_in_last_7 >= 5) {
            negDrivers.push(`😓 <strong>Heavy Schedule</strong>: ${impact} pts (${fb.games_in_last_7} games in 7 days)`);
        }
    }
    
        if (fb.efficiency_adj && Math.abs(fb.efficiency_adj) > 0.15) {
            const impact = fb.efficiency_adj.toFixed(2);
            if (fb.efficiency_adj > 0) {
                posDrivers.push(`🔥 <strong>Shooting Efficiency</strong>: +${impact} pts (hot streak)`);
            } else {
                negDrivers.push(`🧊 <strong>Shooting Efficiency</strong>: ${impact} pts (cold streak)`);
            }
        }
        
        if (fb.momentum_adj && Math.abs(fb.momentum_adj) > 0.15) {
            const impact = fb.momentum_adj.toFixed(2);
            if (fb.momentum_adj > 0) {
                posDrivers.push(`🏆 <strong>Team Momentum</strong>: +${impact} pts (winning streak)`);
            } else {
                negDrivers.push(`📉 <strong>Team Momentum</strong>: ${impact} pts (team slump)`);
            }
        }
        
        // Injury information (teamContext already defined above)
        if (fb.opp_injury_adj && Math.abs(fb.opp_injury_adj) > 0.1) {
            const impact = fb.opp_injury_adj.toFixed(2);
            let injuryText = `🏥 <strong>Opponent Injuries</strong>: +${impact} pts`;
            
            if (oppInjuries && oppInjuries.total_players_out > 0) {
                const activeInjuries = oppInjuries.active_injuries || [];
                if (activeInjuries.length > 0) {
                    const playerNames = activeInjuries.map(inj => inj.player_name || inj.name || 'Unknown');
                    // Show first 3 in factor list, but all are in injury report box
                    const displayNames = playerNames.slice(0, 3);
                    injuryText += ` (${displayNames.join(', ')}${playerNames.length > 3 ? `, +${playerNames.length - 3} more` : ''})`;
                } else {
                    injuryText += ` (${oppInjuries.total_players_out} ${oppInjuries.total_players_out === 1 ? 'player' : 'players'} out)`;
                }
            } else {
                injuryText += ` (weakened defense)`;
            }
            
            posDrivers.push(injuryText);
        }
        
        // Own team injuries (negative factor) - ownInjuries already declared above
        if (ownInjuries && ownInjuries.total_players_out > 0 && fb.injury_impact_adj && Math.abs(fb.injury_impact_adj) > 0.1) {
            const impact = fb.injury_impact_adj.toFixed(2);
            const activeInjuries = ownInjuries.active_injuries || [];
            let injuryText = `🏥 <strong>Team Injuries</strong>: ${impact} pts`;
            
            if (activeInjuries.length > 0) {
                const playerNames = activeInjuries.map(inj => inj.player_name || inj.name || 'Unknown');
                // Show first 3 in factor list
                const displayNames = playerNames.slice(0, 3);
                injuryText += ` (${displayNames.join(', ')}${playerNames.length > 3 ? `, +${playerNames.length - 3} more` : ''} - may increase usage)`;
            } else {
                injuryText += ` (${ownInjuries.total_players_out} teammate${ownInjuries.total_players_out === 1 ? '' : 's'} out)`;
            }
            
            // Team injuries can be positive (more usage) or negative (worse team performance)
            if (fb.injury_impact_adj > 0) {
                posDrivers.push(injuryText);
            } else {
                negDrivers.push(injuryText);
            }
        }
        
        if (fb.dvp_adj && Math.abs(fb.dvp_adj) > 0.2) {
            const impact = fb.dvp_adj.toFixed(2);
            if (fb.dvp_adj > 0) {
                posDrivers.push(`🎯 <strong>Matchup</strong>: +${impact} pts (favorable defense)`);
            } else {
                negDrivers.push(`🛡️ <strong>Matchup</strong>: ${impact} pts (tough defense)`);
            }
        }
        
        if (fb.defender_adj && Math.abs(fb.defender_adj) > 0.2) {
            const impact = fb.defender_adj.toFixed(2);
            negDrivers.push(`🔒 <strong>Primary Defender</strong>: ${impact} pts (elite defender)`);
        }
        
        if (fb.usage_adj && Math.abs(fb.usage_adj) > 0.2) {
            const impact = fb.usage_adj.toFixed(2);
            if (fb.usage_adj > 0) {
                posDrivers.push(`📊 <strong>Usage Rate</strong>: +${impact} pts (high involvement)`);
            }
        }
        
        if (fb.minutes_ratio && fb.minutes_ratio > 1.05) {
            const pct = ((fb.minutes_ratio - 1) * 100).toFixed(0);
            posDrivers.push(`⏱️ <strong>Playing Time</strong>: +${pct}% expected minutes`);
        }
    }
    
    if (posDrivers.length > 0) {
        explanationHTML += `
            <div style="margin-bottom: 1rem;">
                <strong style="color: #10b981; font-size: 1.05rem;">✅ Positive Factors:</strong>
                <ul style="margin-top: 0.5rem; padding-left: 1.5rem; line-height: 1.8;">
                    ${posDrivers.map(d => `<li>${d}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    if (negDrivers.length > 0) {
        explanationHTML += `
            <div style="margin-bottom: 1rem;">
                <strong style="color: #ef4444; font-size: 1.05rem;">⚠️ Negative Factors:</strong>
                <ul style="margin-top: 0.5rem; padding-left: 1.5rem; line-height: 1.8;">
                    ${negDrivers.map(d => `<li>${d}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // 3. Historical Context
    if (fb.empirical_hit_rate != null) {
        const hitRate = (fb.empirical_hit_rate * 100).toFixed(1);
        const hitRateColor = fb.empirical_hit_rate >= 0.6 ? '#10b981' : fb.empirical_hit_rate <= 0.4 ? '#ef4444' : '#f59e0b';
        explanationHTML += `
            <div style="margin-top: 1.5rem; padding: 1rem; background: #f9fafb; border-radius: 0.5rem;">
                <strong style="font-size: 1.05rem;">📊 Historical Performance</strong><br>
                <p style="margin-top: 0.5rem;">Against similar lines, this player has gone over <strong style="color: ${hitRateColor};">${hitRate}%</strong> of the time 
                (${analysis.times_hit || '?'} of ${analysis.total_games || '?'} games).</p>
            </div>
        `;
    }
    
    // 4. Final Recommendation
    const confColor = analysis.confidence === 'HIGH' ? '#10b981' : analysis.confidence === 'MEDIUM' ? '#f59e0b' : '#ef4444';
    explanationHTML += `
        <div style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; border-radius: 0.5rem;">
            <strong style="font-size: 1.1rem;">🎯 Final Recommendation</strong><br>
            <p style="margin-top: 0.5rem;"><strong style="color: ${confColor};">${analysis.recommendation}</strong> with 
            <strong style="color: ${confColor};">${analysis.confidence}</strong> confidence. 
            ${analysis.confidence === 'HIGH' ? 'Strong statistical edge detected.' : 
              analysis.confidence === 'MEDIUM' ? 'Moderate edge with some uncertainty.' : 
              'Factors are conflicting or data is limited.'}</p>
        </div>
    `;
    
    explanationHTML += '</div>';
    el.innerHTML = explanationHTML;
}

function updatePlayerContext(playerContext, stats, factors) {
    const container = document.getElementById('playerContext');
    if (!container) return;
    container.innerHTML = '';
    
    if (playerContext && stats) {
        const propStats = stats['points'] || {};
        const shooting = stats['shooting'] || {};
        const momentum = stats['momentum'] || {};
        const schedule = stats['schedule'] || {};
        
        const items = [
            { label: 'Position', value: playerContext.position || 'N/A' },
            { label: 'Season Avg', value: propStats.avg ? propStats.avg.toFixed(1) : 'N/A' },
            { label: 'Last 5 Avg', value: propStats.last5_avg ? propStats.last5_avg.toFixed(1) : 'N/A' },
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
                { label: 'VS Team Avg', value: matchupHistory.avg_points ? matchupHistory.avg_points.toFixed(1) : 'N/A' },
                { label: 'Previous Games', value: matchupHistory.games_played || 0 },
                { label: 'Success Rate', value: matchupHistory.success_rate ? 
                       `${(matchupHistory.success_rate * 100).toFixed(1)}%` : 'N/A' }
            );
        }
        
        if (positionMatchup) {
            items.push(
                { label: 'Position Defense', value: positionMatchup.defensive_rating ?
                       positionMatchup.defensive_rating.toFixed(1) : 'N/A' },
                { label: 'Pts Allowed', value: positionMatchup.pts_allowed_per_game ?
                       positionMatchup.pts_allowed_per_game.toFixed(1) : 'N/A' }
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
    
    const values = propType.includes('_') ? 
        stats.combined_stats[propType]?.values || [] : 
        stats[propType]?.values || [];
    
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
    
    let values = propType.includes('_') ? 
        stats.combined_stats[propType]?.values || [] : 
        stats[propType]?.values || [];
    
    values.forEach((value, index) => {
        const row = document.createElement('tr');
        const isOver = value > line;
        row.className = isOver ? 'bg-green-50' : 'bg-red-50';
        
        row.innerHTML = `
            <td class="py-3 px-4">${new Date(stats.dates[index]).toLocaleDateString()}</td>
            <td class="py-3 px-4 font-medium">${stats.matchups[index]}</td>
            <td class="py-3 px-4 font-bold ${isOver ? 'text-green-600' : 'text-red-600'}">${value.toFixed(1)}</td>
            <td class="py-3 px-4">
                <span class="badge ${isOver ? 'badge-success' : 'badge-danger'}">
                    ${isOver ? 'OVER' : 'UNDER'}
                </span>
            </td>
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
