# Liqheat OpenAI ML Audit

- Generated: 2026-08-06T21:52:05.802199+00:00
- Model: `gpt-5.5`
- Included files: 44

## Executive decision

**Status:** {'classification': 'OBSERVED', 'status': 'research_only_not_ready_for_strategy_selection_or_financial_claims', 'reason': 'Supplied reports show unresolved label-definition, leakage, event-conditioning, overlapping-sample, calibration, and baseline-selection risks. The best reported squeeze configurations are negative or approximately flat after the stated 14 bps round-trip cost, and the strong-contrarian economic backtest path uses future-derived sweep label columns.'}

**Most likely bottleneck:** {'classification': 'INFERRED', 'bottleneck': 'problem_definition_label_execution_validation_alignment', 'reason': 'Across the audits, weak economic results are more plausibly explained by endpoint-return label mismatch, future-oracle execution risk, event-conditioned evaluation, overlapping windows, missing event-level purge, and insufficient baseline controls than by CatBoost hyperparameters.'}

**Keep CatBoost:** {'classification': 'INFERRED', 'decision': True, 'scope': 'keep_as_a_locked_benchmark_and_ablation_model_not_as_the_next_optimization_focus', 'reason': 'CatBoost models are already present and can serve as controlled baselines. However, topology-only underperforms calendar-only for direction_1h in the supplied ablation, and supplied backtests do not show robust positive net expectancy after costs. Further CatBoost tuning should wait until labels, leakage controls, sampling, calibration, and baseline comparisons are fixed.'}



## Experiment roadmap

### 1. E01_label_provenance_and_squeeze_rebuild — Freeze and reproduce squeeze-event label generation

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'The squeeze-event labels can be rebuilt deterministically from raw or feature inputs without future leakage beyond the declared target windows.'}

**Comparison:** Rebuilt outputs versus existing artifacts by row count, event count, class distribution, id-level label equality, and hash where deterministic.

**Implementation:**
- Provide scripts/research_topology_v2_squeeze_events.py and a small reproducible input sample.
- Rebuild detected_squeeze_events.parquet, squeeze_event_dataset.parquet, and walk_forward_predictions.parquet inputs where applicable.
- Create unit tests for precursor window boundaries, event window boundaries, volatility quantile thresholds, overlapping event clustering, and class assignment.
- Record input hashes, output hashes, row counts, class counts, time ranges, and command parameters.

**Success:** {'row_count_match_rate': 1.0, 'event_count_match_rate': 1.0, 'id_level_label_match_rate': 1.0, 'unit_test_pass_rate': 1.0, 'feature_timestamp_less_than_or_equal_logged_at_rate': 1.0}

### 2. E02_label_alignment_and_edge_case_audit — Cross-tab endpoint direction, first-touch sweep, BOTH, INVALID, and strong-contrarian labels

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Endpoint direction_1h materially disagrees with first-touch sweep labels often enough that it should not be treated as the primary liquidation-topology target without qualification.'}

**Comparison:** Endpoint direction labels versus first-touch sweep labels, and first-touch labels versus strong-contrarian labels where available.

**Implementation:**
- Join liq_topology_v2_ml_labeled.parquet, liq_topology_v2_sweep_labels.parquet, and strong_contrarian labels where available by id.
- For 1h rows, cross-tab direction_1h against sweep_code_1h.
- Report INVALID, NONE, LOWER_FIRST, UPPER_FIRST, and BOTH counts by symbol, timeframe, fold, and volatility decile if volatility features are available.
- For BOTH rows, inspect first-hit timestamp resolution, inter-snapshot interval, and price move size where source data permits.
- Define explicit downstream policy for BOTH and INVALID before model training.

**Success:** {'label_policy_required': True, 'sign_conflict_threshold_for_single_primary_label': 'less_than_20_percent_if_endpoint_direction_is_claimed_as_proxy_for_first_touch', 'BOTH_reporting_required': True, 'INVALID_reporting_required': True}

### 3. E03_duplicate_overlap_and_effective_sample_audit — Measure duplicate timestamps, overlapping windows, event clustering, and non-overlapping metric degradation

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Dense overlapping snapshots and repeated timestamps materially reduce effective independent sample size relative to reported row counts.'}

**Comparison:** Dense row-level metrics versus non-overlapping bucket metrics and block-bootstrap intervals.

**Implementation:**
- Group feature rows by symbol, timeframe, and logged_at to count duplicate timestamp rows.
- Compute duplicate share, duplicate price share, and whether duplicate groups cross train, validation, or test boundaries.
- Compute label autocorrelation and prediction autocorrelation at multiple snapshot lags for each symbol/timeframe.
- Construct non-overlapping horizon buckets and keep one row per symbol/timeframe/bucket.
- Recompute available model metrics on dense rows versus non-overlapping rows.
- Estimate confidence intervals using day-level or symbol-day block bootstrap.

**Success:** {'duplicate_share_after_preprocessing': 0.0, 'metric_degradation_bound': 'less_than_10_percent_relative_drop_or_explicitly_reported_as_unstable', 'reporting_required': ['nominal_row_count', 'non_overlapping_count', 'effective_sample_size_estimate', 'block_bootstrap_intervals']}

### 4. E04_oracle_free_strong_contrarian_execution — Replace label-column execution with causal chronological trigger simulation

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'A causal simulator using only signal-time levels and future observed prices can reproduce the label-driven strong-contrarian trade set; if not, the label-driven backtest is not executable evidence.'}

**Comparison:** Label-driven simulator versus oracle-free chronological simulator.

**Implementation:**
- Remove sweep_code_1h, first_hit_seconds_1h, post_hit_continuation_pct_1h, and strong_contrarian labels from signal selection and entry timing.
- At logged_at, place a virtual far-side trigger using levels known at logged_at.
- Advance through the chronological price stream sequentially.
- Enter only when the observed price crosses the trigger.
- Apply cancellation, stop, target, post-entry window, cooldown, and cost assumptions without using future label columns.
- Compare causal trade ids, entry timestamps, trigger rates, and returns against the existing label-driven simulator.

**Success:** {'for_label_simulator_equivalence_claim': {'trade_id_match_rate': 'at_least_0.99', 'entry_time_match_within_one_snapshot_rate': 'at_least_0.99'}, 'for_economic_reporting': 'all headline metrics must come from the oracle-free simulator'}

### 5. E05_full_stream_event_level_purged_nested_validation — Evaluate squeeze and event models with full-stream scoring, event-level purge, and untouched outer holdout

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Reported squeeze performance changes when evaluation is full-stream, event-level purged, and separated into inner selection and outer final evaluation.'}

**Comparison:** Existing squeeze_signal and full_stream reports versus purged full-stream nested walk-forward results.

**Implementation:**
- Attach stable event keys where event labels exist, such as symbol plus event time plus event direction or other deterministic event identifier.
- Score every chronological 1h topology snapshot in each test fold, not only event-oriented prediction rows.
- Use inner folds for threshold and execution-grid selection.
- Evaluate exactly one frozen selected configuration on an untouched later chronological outer holdout.
- Purge training rows whose label windows overlap validation or test intervals.
- Purge rows sharing the same stable event key across train, validation, and test.

**Success:** {'input_coverage_rate_relative_to_full_1h_stream': 1.0, 'event_key_intersection_count': 0, 'overlapping_label_window_count_across_splits': 0, 'outer_holdout_required': True, 'selected_configuration_count_on_outer_holdout': 1}

### 6. E06_baseline_ladder_and_no_trade_comparator — Establish mandatory no-trade, calendar, symbol, distance-only, and simple-rule baselines

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Some observed model performance can be matched by simpler baselines such as calendar/symbol or distance-only rules.'}

**Comparison:** Full model versus no-trade, base-rate, calendar/symbol, distance-only, nearest-side, and deterministic liquidity-distance baselines.

**Implementation:**
- For each target family, train or compute baselines on identical purged walk-forward folds.
- Include no-trade baseline in every economic summary.
- Include majority or base-rate baseline for classification.
- Include calendar_only, symbol_only, calendar_plus_symbol, topology_only, topology_without_symbol, distance_only, nearest_side, and deterministic distance_advantage threshold baselines where applicable.
- For economic tests, apply the same oracle-free execution assumptions to model and baseline signals.

**Success:** {'topology_increment_for_direction_claim': 'at_least_plus_0.02_balanced_accuracy_over_calendar_plus_symbol_with_blocked_confidence_interval_excluding_0', 'full_model_increment_for_sweep_claim': 'at_least_plus_0.02_balanced_accuracy_or_plus_0.03_ROC_AUC_over_distance_only', 'economic_selection_gate': 'must_beat_no_trade_after_costs_on_pre_registered_outer_evaluation_before_any_strategy_claim'}

### 7. E07_probability_calibration_and_selective_prediction — Audit probability calibration, confidence thresholds, and abstention behavior

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Raw model probabilities and direction_confidence are not sufficiently calibrated or monotonic to support fixed confidence thresholds without validation-fold calibration.'}

**Comparison:** Raw probabilities versus validation-calibrated probabilities, and low-confidence versus high-confidence cohorts.

**Implementation:**
- Compute reliability curves, ECE, MCE, Brier score, log_loss, and bucketed event rates for each fold.
- Report bucketed metrics by symbol, timeframe, and confidence decile.
- Compare raw probabilities with validation-only temperature scaling, isotonic calibration, and Platt calibration where applicable.
- Plot or tabulate coverage, event rate, direction accuracy, and mean net bps by probability and confidence threshold.
- Require thresholds to be selected only on validation or inner folds.

**Success:** {'calibration_improvement': 'ECE_reduction_at_least_30_percent_without_ROC_AUC_or_PR_AUC_drop_greater_than_0.005', 'threshold_transfer': 'test_alert_count_within_plus_or_minus_25_percent_of_validation_expected_count_per_fold', 'confidence_filter_requirement': 'higher_confidence_buckets_must_show_statistically_higher_event_or_direction_quality_than_lower_buckets_with_minimum_sample_counts'}

### 8. E08_event_to_execution_diagnostics — Measure whether event labels imply tradable post-cost paths

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'High event precision and direction accuracy fail economically because events are too late, too small, too volatile, or poorly aligned with stop and target geometry after costs.'}

**Comparison:** Event-detected versus no-event alerts, direction-correct versus direction-incorrect alerts, and alternative pre-registered exit geometries.

**Implementation:**
- For every alert, compute post-alert and post-entry MFE, MAE, time to favorable excursion, time to adverse excursion, stop hit, target hit, and net bps after stated costs.
- Stratify by true event class, predicted class, direction correctness, lead time, probability bucket, symbol, timeframe, and fold.
- Compare event-positive/economic-positive rows against event-positive/economic-negative rows.
- Test fixed target/stop, nearest-liquidity target/stop, and volatility-scaled exits only under oracle-free simulation.

**Success:** {'diagnostic_requirement': 'all selected event labels must show economic separation between event_positive_economic_positive and event_positive_economic_negative cohorts', 'strategy_promotion_gate': 'positive post-cost outer-holdout metrics required before any strategy claim', 'no_promise': 'classification lift alone is insufficient'}

### 9. E09_group_regime_and_symbol_generalization — Test stability by symbol, timeframe, volatility, trend, and matrix/topology regime

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Aggregate performance hides unstable behavior across symbols, timeframes, folds, and regimes.'}

**Comparison:** Aggregate model performance versus worst-group, macro-group, leave-one-symbol-out, and leave-one-timeframe-out performance.

**Implementation:**
- Report metrics by fold, symbol, timeframe, symbol_timeframe, and volatility decile.
- If matrix/regime features are available with verified timestamp alignment, add regime cohorts such as trend alignment, topology/matrix agreement, and conflict.
- Run leave-one-symbol-out and leave-one-timeframe-out validation for primary targets.
- Compute worst-group metrics and dispersion, not only aggregate averages.
- Mark unsupported groups if sample counts or validation metrics fail pre-registered gates.

**Success:** {'minimum_group_reporting': 'fold_by_symbol_by_timeframe_required', 'deployment_scope_rule': 'groups_failing_minimum_sample_or_metric_gates_must_be_marked_unsupported', 'leave_group_out_requirement': 'topology_claims_require_positive_incremental_lift_over_baseline_on_group_held_out_tests'}

### 10. E10_dynamic_and_volatility_normalized_topology_features — Add backward-only dynamic topology and volatility-normalized feature blocks

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Temporal topology behavior and volatility-normalized distances add more transferable signal than static raw distances and log volumes alone.'}

**Comparison:** Static topology versus dynamic topology, volatility-normalized topology, dynamic plus normalized topology, and full feature set.

**Implementation:**
- Create backward-only features such as level_age_minutes, nearest_side_stability_minutes, distance_velocity, pool_volume_velocity, active_level_velocity, prior_sweep_count, post_sweep_elapsed_minutes, and cluster_persistence_score if source data supports them.
- Create volatility-normalized features such as distance_to_recent_realized_volatility, distance_to_ATR, liquidity_pressure_per_volatility, and volume relative to rolling symbol/timeframe baselines if source data supports them.
- Use a feature-family registry to prevent uncontrolled raw/log/ratio duplication.
- Evaluate blocks under identical purged walk-forward folds after baseline experiments are complete.

**Success:** {'direction_or_sweep': 'at_least_plus_0.02_balanced_accuracy_over_static_topology', 'strong_contrarian': 'at_least_plus_0.02_PR_AUC_or_plus_10_percent_relative_top_5pct_lift_over_current_static_full_combined_baseline', 'feature_stability': 'top_15_feature_rank_correlation_greater_than_0.5_across_folds_if_feature_importance_claims_are_made'}

### 11. E11_two_stage_edge_direction_modeling — Compare multiclass softmax against explicit event model plus conditional direction model

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'Separating event probability from event-conditioned direction improves calibration or direction quality at matched alert coverage versus a single multiclass model.'}

**Comparison:** Single multiclass model versus two-stage event plus conditional direction model at matched alert coverage.

**Implementation:**
- Train Stage 1 as event/no-event or economic-edge/no-edge model.
- Train Stage 2 as conditional direction model on event-positive or valid first-touch rows only.
- Generate out-of-fold probabilities for both stages.
- Compare against existing multiclass squeeze probability and direction_confidence construction.
- Apply abstention based on both event probability and conditional direction confidence selected only on validation folds.

**Success:** {'matched_coverage_requirement': True, 'success_condition': 'two_stage_model_improves_event_conditioned_direction_accuracy_or_top_decile_event_lift_at_matched_coverage_and_has_lower_ECE_than_multiclass_baseline', 'economic_condition': 'economic metrics_must_be_reported_but_no_financial_performance_promise_is_allowed'}

### 12. E12_locked_catboost_and_hyperparameter_holdout_test — Only after prior gates, run limited CatBoost and model-family comparison on frozen targets

**Hypothesis:** {'classification': 'HYPOTHESIS', 'statement': 'After labels, leakage controls, sampling, baselines, and calibration are fixed, limited model tuning may improve locked target metrics without changing research conclusions through data snooping.'}

**Comparison:** Locked CatBoost baseline versus limited pre-registered tuned CatBoost or alternative model families.

**Implementation:**
- Freeze primary target, feature blocks, split protocol, metrics, and selection rule.
- Compare locked CatBoost baseline to a small pre-registered set of model families or CatBoost parameter variants.
- Use inner folds only for hyperparameter choice.
- Evaluate one chosen configuration once on untouched outer holdout.
- Report all tried configurations.

**Success:** {'selection_protocol': 'one_pre_registered_model_selected_inside_inner_folds_and_evaluated_once_on_outer_holdout', 'classification_improvement': 'blocked_confidence_interval_for_incremental_metric_excludes_0', 'economic_claim_gate': 'post_cost_outer_holdout_economic_metrics_required_but_not_promised'}

## Individual audits

### problem_and_label_audit

The supplied files show several label and horizon issues that should be audited before hyperparameter work. The strongest observed issues are: direction_1h is a fixed endpoint-return label rather than a first-touch liquidity-sweep label; training rows are dense overlapping snapshots; calendar/symbol features dominate the 1h direction task; and high event/direction metrics in squeeze backtests do not translate into positive net bps under the reported assumptions. Sweep-label code does implement UPPER_FIRST, LOWER_FIRST, NONE, BOTH and INVALID logic, but downstream treatment of BOTH/INVALID and repeated timestamp independence need explicit measurement. Evidence for the original squeeze-event label builder is incomplete because scripts/research_topology_v2_squeeze_events.py is referenced but not supplied.

### leakage_and_validation_audit

The supplied files show several validation risks. The most severe observed issue is a future-oracle economic backtest in the strong contrarian pipeline: execution uses sweep labels and first-hit timing that are only knowable after the signal time. The squeeze signal backtest is also materially weaker than the full-stream backtest because it operates on walk-forward prediction rows rather than every chronological snapshot, and its reported event_rate is 1.0 for top configurations. Model reports show strong dependence on symbol/calendar features, weak incremental topology value for the 1h direction task, heavy grid search over the same OOS folds, overlapping time-series observations, missing calibration evidence, and misleading ranking where selected configurations can have negative net expectancy. Some safeguards are present, including chronological splits, stated embargoes, and explicit forbidden feature patterns, but row-level evidence proving event-level purge, no duplicate-event overlap, calibration stability, and symbol/timeframe generalization is missing.

### feature_and_model_audit

The supplied evidence shows that the current 1h direction CatBoost topology model is not yet feature-led: calendar and symbol features dominate, topology-only underperforms calendar-only on the held-out test split, and the combined ablation stops at iteration 1 with most topology features at zero importance. In contrast, contrarian-sweep and strong-contrarian targets show materially stronger classification lift, especially in top-fraction ranking, but the economic backtests shown for squeeze configurations are negative or approximately breakeven after 14 bps round-trip costs. The main audit priority is not hyperparameter search; it is target/feature/execution alignment: add measurable temporal liquidity-state features, volatility/distance normalization, calibration diagnostics, regime and multi-timeframe alignment tests, redundancy controls, and oracle-safe execution simulation before expanding model tuning.

### adversarial_reviewer

The supplied evidence does not support the premise that weak economic performance is mainly a CatBoost modeling problem. For the 1h direction target, topology features underperform a simple calendar/symbol baseline on test, while CatBoost plus all features is only marginally above calendar-only in one report and below calendar-only in the ablation. Trading backtests show that high event/direction metrics do not translate into positive net execution after the stated 14 bps round-trip cost; the reported best full-stream and squeeze-signal configurations are negative or approximately flat, making no-trade a valid baseline that is not treated as the primary comparator. The more promising strong-contrarian target shows ranking lift and walk-forward AUC, but absolute precision remains low and economic robustness is unknown from the supplied output files. The main adversarial conclusion is that target definition, stationarity, effective sample size, and selection bias are the first-order risks; model choice is secondary until these are falsified.
