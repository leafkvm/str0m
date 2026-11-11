use std::cmp::max;
use std::cmp::min;
use std::time::{Duration, Instant};

use super::macros::log_inherent_loss;
use super::macros::log_loss_based_bitrate_estimate;
use super::macros::log_loss_bw_limit_in_window;
use crate::rtp_::TwccSendRecord;
use crate::{Bitrate, DataSize};

use super::time::{TimeDelta, Timestamp};

/// Loss controller based on libWebRTC's `LossBasedBweV2`.
///
/// ## Overview
///
/// The estimator attempts to estimate the inherent loss of the link using Maximum Likelihood
/// Estimation of an assumed Bernoulli distribution. This allows it to distinguish congestion
/// induced loss from this inherent loss.
///
/// The controller integrates with ALR (Application Limited Region) detection and link capacity
/// tracking. When in ALR, it uses proven link capacity from successful ALR probes as an upper
/// bound on estimates, preventing overestimation in application-limited scenarios. When
/// transitioning into or out of ALR, the controller resets its observation window to avoid
/// mixing traffic patterns from different network utilization regimes.
///
/// The estimate is bounded by the output of the delay-based estimator, meaning this controller
/// can only reduce estimates (acting as a safety cap), not increase them.
///
///
/// Ref:
/// * https://webrtc.googlesource.com/src/+/main/modules/congestion_controller/goog_cc/loss_based_bwe_v2.cc
/// * https://webrtc.googlesource.com/src/+/main/modules/congestion_controller/goog_cc/loss_based_bwe_v2.h
pub struct LossController {
    /// Configuration for the controller.
    config: Config,

    /// The current state of the controller.
    state: LossControllerState,

    /// Staging ground for observations while they are being constructed.
    partial_observation: PartialObservation,

    /// The last packet sent in the most recent observation.
    last_send_time_most_recent_observation: Timestamp,

    // Observation window
    /// Forever growing counter of observations. Observation::id derives from this.
    num_observations: u64,
    /// Window of observations.
    observations: Box<[Observation]>,
    /// Temporal weights, used to weight observations by recency. Same size as `observations`.
    temporal_weights: Box<[f64]>,
    /// Upper bound temporal weights, used to weight observations by recency. Same size as `observations`.
    instant_upper_bound_temporal_weights: Box<[f64]>,

    /// Precomputed instantaneous upper bound on bandwidth estimate.
    cached_instant_upper_bound: Option<Bitrate>,
    /// Last time we reduced the estimate.
    last_time_estimate_reduced: Timestamp,

    /// When we started recovering after being loss limited last time.
    /// While in this window the bandwidth estimate is bounded by `bandwidth_limit_in_current_window`.
    recovering_after_loss_timestamp: Timestamp,
    /// Upper bound on estimate while in recovery window.
    bandwidth_limit_in_current_window: Bitrate,

    /// The current estimate
    current_estimate: ChannelParameters,

    /// The min bitrate we will emit as an estimate.
    min_bitrate: Bitrate,
    /// The max bitrate we will emit as an estimate.
    max_bitrate: Bitrate,

    /// The most recent acknowledged bitrate derived from TWCC.
    acknowledged_bitrate: Bitrate,

    /// The most recent estimated bitrate from the delay based estimator.
    delay_based_estimate: Bitrate,

    /// HOLD mechanism state - prevents immediate ramp-up after loss
    last_hold_info: HoldInfo,

    /// The estimate that started the current padding duration, and when (libWebRTC's
    /// `last_padding_info_`).
    last_padding_info: PaddingInfo,

    /// ALR start time (None if not in ALR)
    alr_start_time: Option<Instant>,

    /// Link capacity estimate from probes during ALR
    link_capacity_estimate: Option<Bitrate>,

    /// Previous ALR state to detect transitions
    was_in_alr: bool,
}

/// State of the Loss Controller
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum LossControllerState {
    /// LossController is in increasing state
    Increasing,
    /// LossController is in increasing state and the pacer pads up to the estimate, so that
    /// the acknowledged bitrate, which bounds the increase, can follow the estimate.
    IncreaseUsingPadding,
    /// LossController is in decreasing state
    Decreasing,
    /// LossController is in relaying the estimate of the delay controller
    DelayBased,
}

pub trait PacketResult {
    /// When the packet was sent
    fn local_send_time(&self) -> Instant;
    /// Size of the packet payload
    fn size(&self) -> DataSize;

    /// Whether this packet was lost or not.
    fn lost(&self) -> bool;
}

impl LossController {
    pub fn new() -> LossController {
        let config = Config::default();

        let mut controller = LossController {
            state: LossControllerState::DelayBased,
            partial_observation: PartialObservation::new(),
            last_send_time_most_recent_observation: Timestamp::DistantFuture,
            observations: vec![Observation::DUMMY; config.observation_window_size]
                .into_boxed_slice(),
            num_observations: 0,
            temporal_weights: vec![0_f64; config.observation_window_size].into_boxed_slice(),
            instant_upper_bound_temporal_weights: vec![0_f64; config.observation_window_size]
                .into_boxed_slice(),
            cached_instant_upper_bound: None,
            last_time_estimate_reduced: Timestamp::DistantPast,
            recovering_after_loss_timestamp: Timestamp::DistantPast,
            bandwidth_limit_in_current_window: Bitrate::MAX,

            current_estimate: ChannelParameters::new(config.initial_inherent_loss_estimate),

            min_bitrate: Bitrate::kbps(1),
            max_bitrate: Bitrate::INFINITY,

            // review usage from here on after
            acknowledged_bitrate: Bitrate::INFINITY,
            delay_based_estimate: Bitrate::INFINITY,

            last_hold_info: HoldInfo::default(),
            last_padding_info: PaddingInfo::default(),

            alr_start_time: None,
            link_capacity_estimate: None,
            was_in_alr: false,

            config,
        };

        // Initialize weights
        {
            let this = &mut controller;
            for i in 0..this.config.observation_window_size {
                let val = f64::powi(this.config.temporal_weight_factor, i as i32);
                this.temporal_weights[i] = val;
                let val = f64::powi(
                    this.config.instant_upper_bound_temporal_weight_factor,
                    i as i32,
                );
                this.instant_upper_bound_temporal_weights[i] = val;
            }
        };

        controller
    }

    /// Override the current bandwidth estimate.
    pub fn set_bandwidth_estimate(&mut self, bandwidth_estimate: Bitrate) {
        self.current_estimate.loss_limited_bandwidth = bandwidth_estimate;
    }

    /// Update the acknowledged bitrate based on TWCC feedback.
    pub fn set_acknowledged_bitrate(&mut self, acknowledged_bitrate: Bitrate) {
        self.acknowledged_bitrate = acknowledged_bitrate;
    }

    /// Set ALR start time from the ALR detector.
    pub fn set_alr_start_time(&mut self, alr_start: Option<Instant>) {
        let was_in_alr = self.was_in_alr;
        let is_in_alr = alr_start.is_some();

        // Detect ALR state transition
        if was_in_alr != is_in_alr {
            // Reset observations on ALR transition to avoid mixing
            // ALR and non-ALR traffic in the same observation window
            self.reset_observations();
            trace!(
                "LossController: ALR state changed (was: {}, now: {}), observations reset",
                was_in_alr, is_in_alr
            );
        }

        self.alr_start_time = alr_start;
        self.was_in_alr = is_in_alr;
    }

    /// Set link capacity estimate from successful ALR probes.
    pub fn set_link_capacity_estimate(&mut self, capacity: Option<Bitrate>) {
        self.link_capacity_estimate = capacity;
    }

    /// Check if currently in ALR state
    fn is_in_alr(&self) -> bool {
        self.alr_start_time.is_some()
    }

    /// Reset all observations.
    ///
    /// Called when ALR state transitions to avoid mixing traffic patterns
    /// from different network utilization regimes.
    fn reset_observations(&mut self) {
        self.partial_observation = PartialObservation::new();
        self.last_send_time_most_recent_observation = Timestamp::DistantFuture;

        // Clear observation window
        for observation in self.observations.iter_mut() {
            *observation = Observation::DUMMY;
        }

        // Reset cached values that depend on observations
        self.cached_instant_upper_bound = None;
    }

    /// Update the estimate using TWCC feedback from the network.
    /// After this [`loss_based_result`] returns the latest estimate.
    pub fn update_bandwidth_estimate(
        &mut self,
        packet_results: &[impl PacketResult],
        delay_based_estimated: Bitrate,
    ) {
        self.delay_based_estimate = delay_based_estimated;

        if packet_results.is_empty() {
            debug!("packet results is empty");
            return;
        }

        if !self.maybe_add_observation(packet_results) {
            return;
        }

        if !self.current_estimate.loss_limited_bandwidth.is_valid() {
            warn!("estimator must be initialized before use");
            return;
        }

        let mut best_candidate = self.current_estimate;
        let mut objective_max = f64::MIN;

        for candidate in self.get_candidates().iter_mut() {
            self.newtons_method_update(candidate);

            let candidate_objective = self.get_objective(candidate);
            if candidate_objective > objective_max {
                objective_max = candidate_objective;
                best_candidate = *candidate;
            }
        }

        if best_candidate.loss_limited_bandwidth < self.current_estimate.loss_limited_bandwidth {
            self.last_time_estimate_reduced = self.last_send_time_most_recent_observation;
        }

        // do not increase the estimate if the average loss is greater than current inherent loss
        if self.average_reported_loss_ratio() > best_candidate.inherent_loss
            && self
                .config
                .not_increase_if_inherent_loss_less_than_average_loss
            && self.current_estimate.loss_limited_bandwidth < best_candidate.loss_limited_bandwidth
        {
            best_candidate.loss_limited_bandwidth = self.current_estimate.loss_limited_bandwidth;
        }

        if self.is_bandwidth_limited_due_to_loss() {
            // Bound the estimate increase if:
            // 1. The estimate has been increased for less than
            // `delayed_increase_window` ago, and
            // 2. The best candidate is greater than bandwidth_limit_in_current_window.

            if self.recovering_after_loss_timestamp.is_exact()
                && self.recovering_after_loss_timestamp + self.config.delayed_increase_window
                    > self.last_send_time_most_recent_observation
                && best_candidate.loss_limited_bandwidth > self.bandwidth_limit_in_current_window
            {
                best_candidate.loss_limited_bandwidth = self.bandwidth_limit_in_current_window;
            }

            let increase_when_loss_limited =
                self.is_estimate_increasing_when_loss_limited(best_candidate);

            if increase_when_loss_limited && self.acknowledged_bitrate.is_valid() {
                // Choose rampup factor based on whether we're in HOLD region (WebRTC lines 270-281)
                let rampup_factor = if self.last_hold_info.rate.is_valid()
                    && self.acknowledged_bitrate
                        < self.last_hold_info.rate * self.config.bandwidth_rampup_hold_threshold
                {
                    self.config.bandwidth_rampup_upper_bound_factor_in_hold // 1.2
                } else {
                    self.config.bandwidth_rampup_upper_bound_factor // 1.5
                };

                best_candidate.loss_limited_bandwidth =
                    self.current_estimate.loss_limited_bandwidth.max(
                        best_candidate
                            .loss_limited_bandwidth
                            .min(self.acknowledged_bitrate * rampup_factor),
                    );

                // WebRTC lines 282-290: Ensure at least 1 bps increase when transitioning from Decreasing
                if self.state == LossControllerState::Decreasing
                    && best_candidate.loss_limited_bandwidth
                        == self.current_estimate.loss_limited_bandwidth
                {
                    best_candidate.loss_limited_bandwidth =
                        self.current_estimate.loss_limited_bandwidth + Bitrate::bps(1);
                }
            }
        }

        // Never below what the path demonstrably delivered (WebRTC: lower bound
        // by acked rate).
        best_candidate.loss_limited_bandwidth = best_candidate
            .loss_limited_bandwidth
            .max(self.get_instant_lower_bound());

        let loss_limited_bandwidth = best_candidate.loss_limited_bandwidth;

        // HOLD check (WebRTC lines 321-334): If in Decreasing state and HOLD timer active, cap at HOLD rate
        if self.state == LossControllerState::Decreasing
            && self.last_hold_info.timestamp > self.last_send_time_most_recent_observation
            && loss_limited_bandwidth < self.delay_based_estimate
        {
            // Ensure that the acked rate is the lower bound of the HOLD rate.
            self.last_hold_info.rate = self.last_hold_info.rate.max(self.get_instant_lower_bound());
            // During HOLD period, cap estimate at HOLD rate
            self.current_estimate = best_candidate;
            self.current_estimate.loss_limited_bandwidth =
                loss_limited_bandwidth.min(self.last_hold_info.rate);
            log_inherent_loss!(self.current_estimate.inherent_loss);
            log_loss_based_bitrate_estimate!(self.current_estimate.loss_limited_bandwidth.as_f64());
            return;
        }

        // State transitions with HOLD mechanism (WebRTC lines 336-378)
        let bounded_bandwidth_estimate = self.bound_estimate(loss_limited_bandwidth);
        let new_state = if self.is_estimate_increasing_when_loss_limited(best_candidate)
            && self.can_keep_increasing_state(bounded_bandwidth_estimate)
            && loss_limited_bandwidth < delay_based_estimated
            && loss_limited_bandwidth < self.max_bitrate
        {
            let padding_duration = self.config.padding_duration;
            if !padding_duration.is_zero()
                && bounded_bandwidth_estimate > self.last_padding_info.padding_rate
            {
                // Start a new padding duration.
                self.last_padding_info = PaddingInfo {
                    padding_rate: bounded_bandwidth_estimate,
                    padding_timestamp: self.last_send_time_most_recent_observation,
                };
            }
            if padding_duration.is_zero() {
                LossControllerState::Increasing
            } else {
                LossControllerState::IncreaseUsingPadding
            }
        } else if loss_limited_bandwidth < self.delay_based_estimate
            && loss_limited_bandwidth < self.max_bitrate
        {
            // Entering Decreasing state - set HOLD info
            if self.state != LossControllerState::Decreasing
                && self.config.hold_duration_factor > 0.0
            {
                const MAX_HOLD_DURATION: Duration = Duration::from_secs(60);
                self.last_hold_info = HoldInfo {
                    timestamp: self.last_send_time_most_recent_observation
                        + self.last_hold_info.duration,
                    duration: MAX_HOLD_DURATION.min(Duration::from_secs_f64(
                        self.last_hold_info.duration.as_secs_f64()
                            * self.config.hold_duration_factor,
                    )),
                    rate: loss_limited_bandwidth,
                };
            }
            self.last_padding_info = PaddingInfo::default();
            LossControllerState::Decreasing
        } else {
            // Reset HOLD info when returning to DelayBased
            self.last_hold_info = HoldInfo {
                timestamp: Timestamp::DistantPast,
                duration: Duration::from_millis(300),
                rate: Bitrate::INFINITY,
            };
            self.last_padding_info = PaddingInfo::default();
            LossControllerState::DelayBased
        };
        self.set_state(new_state);

        self.current_estimate = best_candidate;
        log_inherent_loss!(self.current_estimate.inherent_loss);
        log_loss_based_bitrate_estimate!(self.current_estimate.loss_limited_bandwidth.as_f64());

        const CONGESTION_CONTROLLER_MIN_BITRATE: Bitrate = Bitrate::kbps(5);
        const CONF_MAX_INCREASE_FACTOR: f64 = 1.3;

        if self.is_bandwidth_limited_due_to_loss()
            && (!self.recovering_after_loss_timestamp.is_exact()
                || self.recovering_after_loss_timestamp + self.config.delayed_increase_window
                    < self.last_send_time_most_recent_observation)
        {
            self.bandwidth_limit_in_current_window = CONGESTION_CONTROLLER_MIN_BITRATE
                .max(loss_limited_bandwidth * CONF_MAX_INCREASE_FACTOR);

            self.recovering_after_loss_timestamp = self.last_send_time_most_recent_observation;
            log_loss_bw_limit_in_window!(self.bandwidth_limit_in_current_window.as_f64());
        }
    }

    // TODO: Determine if we want to integrate these two with the rest of the system.
    #[cfg(test)]
    pub fn set_max_bitrate(&mut self, max_bitrate: Bitrate) {
        self.max_bitrate = max_bitrate;
    }

    #[cfg(test)]
    pub fn set_min_bitrate(&mut self, min_bitrate: Bitrate) {
        self.min_bitrate = min_bitrate;
    }

    pub fn loss_based_result(&self) -> LossBasedBweResult {
        let mut result = LossBasedBweResult {
            bandwidth_estimate: self.current_estimate.loss_limited_bandwidth.as_valid(),
            state: self.state,
        };

        if self.num_observations == 0 {
            return result;
        }

        let Some(loss_limited_bandwidth) = self.current_estimate.loss_limited_bandwidth.as_valid()
        else {
            return result;
        };
        result.bandwidth_estimate = Some(self.bound_estimate(loss_limited_bandwidth));

        result
    }

    /// Bound a loss limited bandwidth by the delay-based estimate and the instant bounds.
    fn bound_estimate(&self, loss_limited_bandwidth: Bitrate) -> Bitrate {
        let instant_upper_bound = self.get_instant_upper_bound();
        let instant_lower_bound = self.get_instant_lower_bound();

        if self.delay_based_estimate.is_valid() {
            loss_limited_bandwidth
                .min(self.delay_based_estimate)
                .min(instant_upper_bound)
                .max(instant_lower_bound)
        } else {
            loss_limited_bandwidth
                .min(instant_upper_bound)
                .max(instant_lower_bound)
        }
    }

    fn maybe_add_observation(&mut self, packet_results: &[impl PacketResult]) -> bool {
        let Some(summary) = PacketResultsSummary::from(packet_results) else {
            return false;
        };

        let last_send_time = Timestamp::from(summary.last_send_time);

        self.partial_observation.update(summary);

        if !self.last_send_time_most_recent_observation.is_exact() {
            self.last_send_time_most_recent_observation = last_send_time;
        }

        let observation_duration = last_send_time - self.last_send_time_most_recent_observation;

        if observation_duration <= Duration::ZERO {
            return false;
        }

        // decide if we can accept the partial observation as complete
        if observation_duration <= self.config.observation_duration_lower_bound {
            return false;
        }

        self.last_send_time_most_recent_observation = last_send_time;

        let observation = {
            let id = self.num_observations;
            self.num_observations += 1;

            Observation {
                num_packets: self.partial_observation.num_packets,
                size: self.partial_observation.size,
                num_lost_packets: self.partial_observation.num_lost_packets,
                lost_size: self.partial_observation.lost_size,
                num_received_packets: self.partial_observation.num_packets
                    - self.partial_observation.num_lost_packets,
                sending_rate: self.partial_observation.size / observation_duration,
                id,
                is_initialized: true,
            }
        };

        // save our complete observation
        self.observations[observation.id as usize % self.config.observation_window_size] =
            observation;

        // renew the partial observation
        self.partial_observation = PartialObservation::new();

        // calculate upper bound
        self.cached_instant_upper_bound = Some(self.calculate_instant_upper_bound());

        true
    }

    fn get_candidates(&self) -> Vec<ChannelParameters> {
        let mut bandwidths = vec![];

        let current = self.current_estimate.loss_limited_bandwidth;

        for factor in self.config.candidate_factor.iter() {
            bandwidths.push(factor * current.as_f64());
        }

        if self.delay_based_estimate.is_valid()
            && self.config.append_delay_based_estimate_candidate
            && self.delay_based_estimate > current
        {
            bandwidths.push(self.delay_based_estimate.as_f64());
        }

        let candidate_bandwidth_upper_bound = self.get_candidate_bandwidth_upper_bound().as_f64();

        if self.config.append_acknowledged_rate_candidate && self.acknowledged_bitrate.is_valid() {
            bandwidths.push(
                (self.acknowledged_bitrate * self.config.bandwidth_backoff_lower_bound_factor)
                    .as_f64(),
            );
        }

        if self.config.append_delay_based_estimate_candidate
            && self.delay_based_estimate.is_valid()
            && self.delay_based_estimate > current
        {
            bandwidths.push(
                (self.delay_based_estimate * self.config.bandwidth_backoff_lower_bound_factor)
                    .as_f64(),
            );
        }

        let mut candidates = Vec::with_capacity(bandwidths.len());

        for bandwidth in bandwidths.iter_mut() {
            let mut candidate = self.current_estimate;
            candidate.loss_limited_bandwidth = if self.config.trendline_integration_enabled {
                bandwidth.min(candidate_bandwidth_upper_bound).into()
            } else {
                bandwidth
                    .min(
                        self.current_estimate
                            .loss_limited_bandwidth
                            .as_f64()
                            .max(candidate_bandwidth_upper_bound),
                    )
                    .into()
            };
            candidate.inherent_loss = self.get_feasible_inherent_loss(&candidate);
            candidates.push(candidate);
        }

        candidates
    }

    fn newtons_method_update(&self, channel_parameters: &mut ChannelParameters) {
        if self.num_observations == 0 {
            return;
        }

        for _ in 0..self.config.newton_iterations {
            let derivatives = self.get_derivatives(channel_parameters);
            channel_parameters.inherent_loss -=
                self.config.newton_step_size * (derivatives.0 / derivatives.1);
            channel_parameters.inherent_loss = self.get_feasible_inherent_loss(channel_parameters);
        }
    }

    fn get_derivatives(&self, channel_prameters: &ChannelParameters) -> (f64, f64) {
        let mut derivatives: (f64, f64) = (0.0, 0.0);

        for observation in self.observations.iter() {
            if !observation.is_initialized {
                continue;
            }

            let loss_probability = self.get_loss_probability(
                channel_prameters.inherent_loss,
                channel_prameters.loss_limited_bandwidth,
                observation.sending_rate,
            );

            let index = (self.num_observations - 1) - observation.id;
            let temporal_weight = self.temporal_weights[index as usize];

            if self.config.use_byte_loss_ratio {
                derivatives.0 += temporal_weight
                    * ((observation.lost_size.as_kb() / loss_probability)
                        - ((observation.size - observation.lost_size).as_kb()
                            / (1.0 - loss_probability)));

                derivatives.1 -= temporal_weight
                    * ((observation.lost_size.as_kb() / f64::powi(loss_probability, 2))
                        + ((observation.size - observation.lost_size).as_kb()
                            / f64::powi(1.0 - loss_probability, 2)));
            } else {
                derivatives.0 += temporal_weight
                    * ((observation.num_lost_packets as f64 / loss_probability)
                        - (observation.num_received_packets as f64 / (1.0 - loss_probability)));

                derivatives.1 -= temporal_weight
                    * ((observation.num_lost_packets as f64 / f64::powi(loss_probability, 2))
                        + (observation.num_received_packets as f64
                            / f64::powi(1.0 - loss_probability, 2)));
            }
        }

        // Clamp second derivative to safe value if invalid due to floating-point edge cases
        // (infinity, denormals, extreme values from pathological TWCC feedback)
        if !derivatives.1.is_sign_negative() || derivatives.1 == 0.0 || derivatives.1.is_nan() {
            derivatives.1 = -1.0e-6;
            debug!(
                "Second derivative clamped to safe value due to invalid result: was {:?}",
                derivatives.1
            );
        }

        derivatives
    }

    fn get_loss_probability(
        &self,
        inherent_loss: f64,
        loss_limited_bandwidth: Bitrate,
        sending_rate: Bitrate,
    ) -> f64 {
        let inherent_loss = inherent_loss.clamp(0.0, 1.0);

        // maybe warn if sending rate or loss limited bandwidth are not finite

        let mut loss_probability = inherent_loss;
        if sending_rate.is_valid()
            && loss_limited_bandwidth.is_valid()
            && sending_rate > loss_limited_bandwidth
        {
            loss_probability += (1.0 - inherent_loss)
                * ((sending_rate - loss_limited_bandwidth).as_f64() / sending_rate.as_f64());
        }

        loss_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    }

    fn get_objective(&self, candidate: &ChannelParameters) -> f64 {
        let mut objective = 0.0;
        let high_bandwidth_bias = self.get_high_bandwidth_bias(candidate.loss_limited_bandwidth);

        for observation in self.observations.iter() {
            if !observation.is_initialized {
                continue;
            }

            let loss_probability = self.get_loss_probability(
                candidate.inherent_loss,
                candidate.loss_limited_bandwidth,
                observation.sending_rate,
            );

            let index = (self.num_observations - 1) - observation.id;
            let temporal_weight = self.temporal_weights[index as usize];

            if self.config.use_byte_loss_ratio {
                objective += temporal_weight
                    * ((observation.lost_size.as_kb() / 1000.0) * f64::ln(loss_probability)
                        + ((observation.size - observation.lost_size).as_kb() / 1000.0)
                            * f64::ln(1.0 - loss_probability));
                objective +=
                    temporal_weight * high_bandwidth_bias * observation.size.as_kb() / 1000.0;
            } else {
                objective += temporal_weight
                    * (observation.num_lost_packets as f64 * f64::ln(loss_probability)
                        + (observation.num_received_packets as f64
                            * f64::ln(1.0 - loss_probability)));

                objective += temporal_weight * high_bandwidth_bias * observation.num_packets as f64;
            }
        }

        objective
    }

    fn is_estimate_increasing_when_loss_limited(&self, candidate: ChannelParameters) -> bool {
        if !self.is_bandwidth_limited_due_to_loss() {
            return false;
        }

        let current = self.current_estimate.loss_limited_bandwidth;
        let candidate = candidate.loss_limited_bandwidth;

        if current < candidate {
            return true;
        }

        current == candidate
            && matches!(
                self.state,
                LossControllerState::Increasing | LossControllerState::IncreaseUsingPadding
            )
    }

    /// Whether `IncreaseUsingPadding` may continue: for `padding_duration` after the
    /// estimate last reached a new high, or while it exceeds that high.
    fn can_keep_increasing_state(&self, estimate: Bitrate) -> bool {
        if self.config.padding_duration.is_zero()
            || self.state != LossControllerState::IncreaseUsingPadding
        {
            return true;
        }

        self.last_padding_info.padding_timestamp + self.config.padding_duration
            >= self.last_send_time_most_recent_observation
            || self.last_padding_info.padding_rate < estimate
    }

    fn is_bandwidth_limited_due_to_loss(&self) -> bool {
        self.state != LossControllerState::DelayBased
    }

    fn get_candidate_bandwidth_upper_bound(&self) -> Bitrate {
        let mut upper_bound = self.max_bitrate;

        // When in ALR and we have a link capacity estimate from probes,
        // use it as the upper bound. This prevents estimating beyond proven capacity.
        if self.is_in_alr() {
            if let Some(capacity) = self.link_capacity_estimate {
                if capacity.is_valid() {
                    upper_bound = upper_bound.min(capacity);
                }
            }
        }

        if self.is_bandwidth_limited_due_to_loss()
            && self.bandwidth_limit_in_current_window.is_valid()
        {
            upper_bound = self.bandwidth_limit_in_current_window.min(upper_bound);
        }

        upper_bound = self.get_instant_upper_bound().min(upper_bound);
        if self.delay_based_estimate.is_valid() {
            upper_bound = upper_bound.min(self.delay_based_estimate);
        }

        if !self.acknowledged_bitrate.is_valid() {
            return upper_bound;
        }

        if self.config.rampup_acceleration_max_factor > Duration::ZERO
            && self.last_send_time_most_recent_observation.is_exact()
            && self.last_time_estimate_reduced.is_exact()
        {
            let delta = (self.last_send_time_most_recent_observation
                - self.last_time_estimate_reduced)
                .max(TimeDelta::ZERO);
            let time_since_bw_reduced = self
                .config
                .rampup_acceleration_maxout_time
                .as_secs_f64()
                .min(delta.as_secs_f64());

            let rampup_acceleration = self.config.rampup_acceleration_max_factor.as_secs_f64()
                * time_since_bw_reduced
                / self.config.rampup_acceleration_maxout_time.as_secs_f64();

            upper_bound = upper_bound + (self.acknowledged_bitrate * rampup_acceleration);
        }

        upper_bound
    }

    fn set_state(&mut self, state: LossControllerState) {
        if state != self.state {
            debug!(
                "Changing loss controller state: {:?} -> {:?}",
                self.state, state
            );
        }
        self.state = state;
    }

    fn get_high_bandwidth_bias(&self, bandwidth: Bitrate) -> f64 {
        if !bandwidth.is_valid() {
            return 0.0;
        }

        let average_reported_loss_ratio = self.average_reported_loss_ratio();

        self.adjust_bias_factor(
            average_reported_loss_ratio,
            self.config.higher_bandwidth_bias_factor,
        ) * bandwidth.as_f64()
            + self.adjust_bias_factor(
                average_reported_loss_ratio,
                self.config.higher_log_bandwidth_bias_factor,
            ) * f64::ln(1.0 + bandwidth.as_f64())
    }

    fn adjust_bias_factor(&self, loss_rate: f64, bias_factor: f64) -> f64 {
        let diff = self.config.threshold_of_high_bandwidth_preference - loss_rate;
        bias_factor * (diff / self.config.bandwidth_preference_smoothing_factor + diff.abs())
    }

    fn calculate_instant_upper_bound(&self) -> Bitrate {
        // this requires someone to set the max bitrate from outside
        let mut instant_limit = self.max_bitrate;

        let average_reported_loss_ratio = self.average_reported_loss_ratio();

        if average_reported_loss_ratio > self.config.instant_upper_bound_loss_offset {
            instant_limit = self.config.instant_upper_bound_bandwidth_balance
                / (average_reported_loss_ratio - self.config.instant_upper_bound_loss_offset);

            if average_reported_loss_ratio > self.config.high_loss_rate_threshold {
                let limit = self.config.bandwidth_cap_at_high_loss_rate
                    - self.config.slope_of_bwe_high_loss_function * average_reported_loss_ratio;

                instant_limit = limit.max(self.min_bitrate);
            }
        }

        instant_limit
    }

    fn get_instant_lower_bound(&self) -> Bitrate {
        let mut lower_bound = Bitrate::ZERO;
        if self.acknowledged_bitrate.is_valid()
            && self.config.lower_bound_by_acked_rate_factor > 0.0
        {
            lower_bound = self.acknowledged_bitrate * self.config.lower_bound_by_acked_rate_factor;
        }
        if self.min_bitrate.is_valid() {
            lower_bound = lower_bound.max(self.min_bitrate);
        }
        lower_bound
    }

    fn get_instant_upper_bound(&self) -> Bitrate {
        self.cached_instant_upper_bound
            .as_valid()
            .unwrap_or(self.max_bitrate)
    }

    fn average_reported_loss_ratio(&self) -> f64 {
        let mut total = 0_f64;
        let mut lost = 0_f64;

        for observation in self.observations.iter() {
            if !observation.is_initialized {
                continue;
            }

            let index = (self.num_observations - 1) - observation.id;

            let instant_temporal_weight = self.instant_upper_bound_temporal_weights[index as usize];

            if self.config.use_byte_loss_ratio {
                total += instant_temporal_weight * observation.size.as_bytes_f64();
                lost += instant_temporal_weight * observation.lost_size.as_bytes_f64();
            } else {
                total += instant_temporal_weight * observation.num_packets as f64;
                lost += instant_temporal_weight * observation.num_lost_packets as f64;
            }
        }

        if total == 0_f64 {
            return 0.0;
        }

        lost / total
    }

    fn get_feasible_inherent_loss(&self, channel_parameters: &ChannelParameters) -> f64 {
        channel_parameters
            .inherent_loss
            .max(self.config.inherent_loss_lower_bound)
            .min(
                self.get_inherent_loss_upper_bound(Some(channel_parameters.loss_limited_bandwidth)),
            )
    }

    fn get_inherent_loss_upper_bound(&self, bandwidth: Option<Bitrate>) -> f64 {
        let Some(bandwidth) = bandwidth else {
            return 1.0;
        };

        if bandwidth == Bitrate::ZERO {
            return 1.0;
        }

        let inherent_loss_upper_bound = self.config.inherent_loss_upper_bound_offset
            + self
                .config
                .inherent_loss_upper_bound_bandwidth_balance
                .as_f64()
                / bandwidth.as_f64();

        inherent_loss_upper_bound.min(1.0)
    }
}

struct Config {
    observation_window_size: usize, // minimum is 2
    observation_duration_lower_bound: Duration,
    trendline_integration_enabled: bool,
    temporal_weight_factor: f64,
    instant_upper_bound_temporal_weight_factor: f64,
    instant_upper_bound_loss_offset: f64,
    instant_upper_bound_bandwidth_balance: Bitrate,
    high_loss_rate_threshold: f64,
    slope_of_bwe_high_loss_function: Bitrate,
    bandwidth_cap_at_high_loss_rate: Bitrate,
    initial_inherent_loss_estimate: f64,
    inherent_loss_upper_bound_offset: f64,
    inherent_loss_upper_bound_bandwidth_balance: Bitrate,
    inherent_loss_lower_bound: f64,
    newton_iterations: usize,
    newton_step_size: f64,
    not_increase_if_inherent_loss_less_than_average_loss: bool,
    delayed_increase_window: Duration,
    bandwidth_rampup_upper_bound_factor: f64,
    candidate_factor: [f64; 3],
    append_acknowledged_rate_candidate: bool,
    append_delay_based_estimate_candidate: bool,
    bandwidth_backoff_lower_bound_factor: f64,
    rampup_acceleration_maxout_time: Duration,
    rampup_acceleration_max_factor: Duration,
    higher_bandwidth_bias_factor: f64,
    higher_log_bandwidth_bias_factor: f64,
    threshold_of_high_bandwidth_preference: f64,
    bandwidth_preference_smoothing_factor: f64,
    use_byte_loss_ratio: bool,
    hold_duration_factor: f64,
    lower_bound_by_acked_rate_factor: f64,
    bandwidth_rampup_hold_threshold: f64,
    bandwidth_rampup_upper_bound_factor_in_hold: f64,
    padding_duration: Duration,
}

#[derive(Debug)]
struct PacketResultsSummary {
    num_packets: u64,
    num_lost_packets: u64,
    total_size: DataSize,
    lost_size: DataSize,
    first_send_time: Instant,
    last_send_time: Instant,
}

impl PacketResultsSummary {
    pub fn new(first_send_time: Instant, last_send_time: Instant) -> PacketResultsSummary {
        PacketResultsSummary {
            num_packets: 0,
            num_lost_packets: 0,
            total_size: DataSize::ZERO,
            lost_size: DataSize::ZERO,
            last_send_time,
            first_send_time,
        }
    }

    pub fn from(records: &[impl PacketResult]) -> Option<PacketResultsSummary> {
        let first = records.first()?;

        let mut summary =
            PacketResultsSummary::new(first.local_send_time(), first.local_send_time());
        for record in records {
            let lost: u64 = record.lost().into();
            let size = record.size();

            summary.num_packets += 1;
            summary.total_size += size;
            summary.lost_size += size * lost;
            summary.num_lost_packets += lost;
            summary.first_send_time = min(summary.first_send_time, record.local_send_time());
            summary.last_send_time = max(summary.last_send_time, record.local_send_time());
        }

        Some(summary)
    }
}

#[derive(Debug, Clone, Copy)]
struct Observation {
    num_packets: u64,
    size: DataSize,
    num_lost_packets: u64,
    lost_size: DataSize,
    num_received_packets: u64,
    sending_rate: Bitrate,
    id: u64,
    is_initialized: bool,
}

impl Observation {
    pub const DUMMY: Self = Self {
        num_packets: 0,
        size: DataSize::ZERO,
        num_lost_packets: 0,
        lost_size: DataSize::ZERO,
        num_received_packets: 0,
        sending_rate: Bitrate::NEG_INFINITY,
        id: 0,
        is_initialized: false,
    };
}

struct PartialObservation {
    num_packets: u64,
    num_lost_packets: u64,
    size: DataSize,
    lost_size: DataSize,
}

impl PartialObservation {
    pub fn new() -> PartialObservation {
        PartialObservation {
            num_packets: 0,
            num_lost_packets: 0,
            size: DataSize::ZERO,
            lost_size: DataSize::ZERO,
        }
    }

    pub fn update(&mut self, summary: PacketResultsSummary) {
        self.num_packets += summary.num_packets;
        self.num_lost_packets += summary.num_lost_packets;
        self.size += summary.total_size;
        self.lost_size += summary.lost_size;
    }
}

/// An estimate derived from some candidate.
#[derive(Debug, Clone, Copy)]
struct ChannelParameters {
    /// The estimated inherent loss
    inherent_loss: f64,
    /// The estimated bandwidth
    loss_limited_bandwidth: Bitrate,
}

impl ChannelParameters {
    pub fn new(inherent_loss: f64) -> ChannelParameters {
        ChannelParameters {
            inherent_loss,
            loss_limited_bandwidth: Bitrate::NEG_INFINITY,
        }
    }
}

/// HOLD mechanism info - prevents immediate ramp-up after loss detection
#[derive(Debug, Clone, Copy)]
struct HoldInfo {
    timestamp: Timestamp,
    duration: Duration,
    rate: Bitrate,
}

impl Default for HoldInfo {
    fn default() -> Self {
        const INIT_HOLD_DURATION: Duration = Duration::from_millis(300);
        Self {
            timestamp: Timestamp::DistantPast,
            duration: INIT_HOLD_DURATION,
            rate: Bitrate::INFINITY,
        }
    }
}

/// The estimate and time at the start of the current padding duration.
#[derive(Debug, Clone, Copy)]
struct PaddingInfo {
    padding_rate: Bitrate,
    padding_timestamp: Timestamp,
}

impl Default for PaddingInfo {
    fn default() -> Self {
        Self {
            padding_rate: Bitrate::NEG_INFINITY,
            padding_timestamp: Timestamp::DistantPast,
        }
    }
}

trait AsValid<T> {
    fn as_valid(&self) -> Option<T>;
}

impl AsValid<Bitrate> for Option<Bitrate> {
    fn as_valid(&self) -> Option<Bitrate> {
        if let Some(bitrate) = self {
            if bitrate.as_f64().is_finite() {
                return Some(*bitrate);
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct LossBasedBweResult {
    pub bandwidth_estimate: Option<Bitrate>,
    pub state: LossControllerState,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            observation_window_size: 15,
            observation_duration_lower_bound: Duration::from_millis(250),
            trendline_integration_enabled: false,
            temporal_weight_factor: 0.9,
            instant_upper_bound_temporal_weight_factor: 0.9,
            instant_upper_bound_loss_offset: 0.05,
            instant_upper_bound_bandwidth_balance: Bitrate::kbps(100),
            high_loss_rate_threshold: 1.0,
            slope_of_bwe_high_loss_function: Bitrate::kbps(1000),
            bandwidth_cap_at_high_loss_rate: Bitrate::kbps(500),
            initial_inherent_loss_estimate: 0.01,
            inherent_loss_upper_bound_offset: 0.05,
            inherent_loss_upper_bound_bandwidth_balance: Bitrate::kbps(100),
            inherent_loss_lower_bound: 1.0e-3,
            newton_iterations: 1,
            newton_step_size: 0.75,
            not_increase_if_inherent_loss_less_than_average_loss: true,
            delayed_increase_window: Duration::from_millis(300),
            bandwidth_rampup_upper_bound_factor: 1.5,
            candidate_factor: [1.02, 1.0, 0.95],
            append_acknowledged_rate_candidate: true,
            append_delay_based_estimate_candidate: true,
            bandwidth_backoff_lower_bound_factor: 1.0,
            rampup_acceleration_maxout_time: Duration::from_secs(60),
            rampup_acceleration_max_factor: Duration::from_secs(60),
            higher_bandwidth_bias_factor: 0.0002,
            higher_log_bandwidth_bias_factor: 0.02,
            threshold_of_high_bandwidth_preference: 0.2,
            bandwidth_preference_smoothing_factor: 0.002,
            use_byte_loss_ratio: true,
            hold_duration_factor: 2.0,
            lower_bound_by_acked_rate_factor: 1.0,
            bandwidth_rampup_hold_threshold: 1.3,
            bandwidth_rampup_upper_bound_factor_in_hold: 1.2,
            padding_duration: Duration::from_secs(2),
        }
    }
}

impl PacketResult for &TwccSendRecord {
    fn local_send_time(&self) -> Instant {
        (*self).local_send_time()
    }

    fn size(&self) -> DataSize {
        (*self).size().into()
    }

    fn lost(&self) -> bool {
        (*self).remote_recv_time().is_none()
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use fastrand::Rng;
    use systemstat::Duration;

    use super::{Bitrate, DataSize, LossBasedBweResult, LossController, LossControllerState};
    use crate::bwe_::loss_recovery_padding_rate;
    use crate::pacer::PacerControl;
    struct PacketResult {
        local_send_time: Instant,
        size: DataSize,
        lost: bool,
    }

    impl super::PacketResult for PacketResult {
        fn local_send_time(&self) -> Instant {
            self.local_send_time
        }

        fn size(&self) -> DataSize {
            self.size
        }

        fn lost(&self) -> bool {
            self.lost
        }
    }

    #[test]
    fn no_loss() {
        // Test no loss, estimate should be bounded by delay based estimate
        let mut lbc = LossController::new();
        lbc.set_min_bitrate(Bitrate::from(50_000)); // 50 kbps
        lbc.set_max_bitrate(Bitrate::from(1_000_000_000)); // 1 Gbps

        let acknowledged_bitrate = Bitrate::from(1_000_000); // 1 Mbps
        lbc.set_acknowledged_bitrate(acknowledged_bitrate);
        lbc.set_bandwidth_estimate(Bitrate::from(1_250_000)); // 1.25Mbps

        let mut pkt_builder = PacketBuilder::new(Instant::now()).num_packets(26);

        // A single observation at 1Mbps
        let result = pkt_builder.build_packets();
        lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
        pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));

        // A single observation at 1Mbps
        let result = pkt_builder.build_packets();
        lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));

        let LossBasedBweResult {
            bandwidth_estimate,
            state,
        } = lbc.loss_based_result();

        assert_eq!(
            bandwidth_estimate,
            Some(Bitrate::bps(1_500_000)),
            "Estimate should increase to delay based estimate, but not further"
        );
        assert_eq!(state, LossControllerState::DelayBased);
    }

    #[test]
    fn stable_loss() {
        // Test stable loss at 5% which should be ignored by the loss controller
        let mut lbc = LossController::new();
        lbc.set_min_bitrate(Bitrate::from(50_000)); // 50 kbps
        lbc.set_max_bitrate(Bitrate::from(1_000_000_000)); // 1 Gbps

        let acknowledged_bitrate = Bitrate::from(1_000_000); // 1 Mbps
        lbc.set_acknowledged_bitrate(acknowledged_bitrate);
        lbc.set_bandwidth_estimate(Bitrate::from(1_250_000)); // 1.25Mbps

        let mut pkt_builder = PacketBuilder::new(Instant::now())
            .with_loss(0.05)
            .num_packets(26);

        // It takes a while for the maximum likelihood estimation to react to the inherent loss
        // this is why we need quite a few observations before the estimate increases to the delay
        // based bound
        // 40 observations(10 seconds) at 1Mbps
        for _ in 0..40 {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        let LossBasedBweResult {
            bandwidth_estimate,
            state,
        } = lbc.loss_based_result();

        assert!(
            state == LossControllerState::DelayBased
                || state == LossControllerState::IncreaseUsingPadding,
            "With stable inherent loss, should be in DelayBased or IncreaseUsingPadding state, got {:?}",
            state
        );

        // Note: The loss controller returns loss_limited_bandwidth even in DelayBased state.
        // The BWE integration layer (SendSideBandwidthEstimator::last_estimate) is responsible
        // for using the delay-based estimate when state is DelayBased.
        // This matches WebRTC's LossBasedBweV2 behavior (see loss_based_bwe_v2.cc:379).
        assert!(
            bandwidth_estimate.is_some(),
            "Loss controller should return an estimate even in DelayBased state"
        );
    }

    #[test]
    fn stable_loss_with_loss_spike() {
        // Test stable loss at 5% which should be ignored by the loss controller, followed by a
        // loss spike which should cause the estimate to dip
        let mut lbc = LossController::new();
        lbc.set_min_bitrate(Bitrate::from(50_000)); // 50 kbps
        lbc.set_max_bitrate(Bitrate::from(1_000_000_000)); // 1 Gbps

        let acknowledged_bitrate = Bitrate::from(1_000_000); // 1 Mbps
        lbc.set_acknowledged_bitrate(acknowledged_bitrate);
        lbc.set_bandwidth_estimate(Bitrate::from(1_250_000)); // 1.25Mbps

        let mut pkt_builder = PacketBuilder::new(Instant::now())
            .with_loss(0.05)
            .num_packets(26);

        // It takes a while for the maximum likelihood estimation to react to the inherent loss
        // this is why we need quite a few observations before the estimate increases to the delay
        // based bound and is stable.
        // 40 observations(10 seconds) at 1Mbps
        for _ in 0..40 {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        pkt_builder = pkt_builder.with_loss(0.9);
        // The receiver only gets 10 % of the 1 Mbps.
        lbc.set_acknowledged_bitrate(Bitrate::kbps(100));
        // Loss spike(1second at 90% loss)
        for _ in 0..4 {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));

            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        let LossBasedBweResult {
            bandwidth_estimate,
            state,
        } = lbc.loss_based_result();

        let estimate = bandwidth_estimate.expect("Should have an estimate");
        assert!(
            estimate < Bitrate::bps(500_000),
            "A loss spike should result in a reduced estimate, estimate was {estimate}"
        );
        assert_eq!(state, LossControllerState::Decreasing);
    }

    #[test]
    fn loss_spike_recovery() {
        // Test stable loss at 5% which should be ignored by the loss controller, followed by a
        // loss spike which should cause the estimate to dip
        let mut lbc = LossController::new();
        lbc.set_min_bitrate(Bitrate::from(50_000)); // 50 kbps
        lbc.set_max_bitrate(Bitrate::from(1_000_000_000)); // 1 Gbps

        let acknowledged_bitrate = Bitrate::from(1_000_000); // 1 Mbps
        lbc.set_acknowledged_bitrate(acknowledged_bitrate);
        lbc.set_bandwidth_estimate(Bitrate::from(1_250_000)); // 1.25Mbps

        let mut pkt_builder = PacketBuilder::new(Instant::now())
            .with_loss(0.05)
            .num_packets(26);

        // It takes a while for the maximum likelihood estimation to react to the inherent loss
        // this is why we need quite a few observations before the estimate increases to the delay
        // based bound and is stable.
        // 40 observations(10 seconds) at 1Mbps
        for _ in 0..40 {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        // Loss spike
        pkt_builder = pkt_builder.with_loss(0.9);
        let result = pkt_builder.build_packets();
        lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
        pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));

        pkt_builder = pkt_builder.with_loss(0.05);
        // Set loss back to 5% and gradually ramp up the bitrate
        for i in 0..40 {
            pkt_builder = pkt_builder.num_packets(6 + i / 2);

            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));

            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        let LossBasedBweResult { state, .. } = lbc.loss_based_result();

        // Note: The loss controller returns loss_limited_bandwidth even in DelayBased state.
        // After recovery, it may be in IncreaseUsingPadding or DelayBased state depending on
        // dynamics. The BWE integration layer uses the delay-based estimate when state is
        // DelayBased.
        assert!(
            state == LossControllerState::DelayBased
                || state == LossControllerState::IncreaseUsingPadding,
            "After recovery from loss spike, should be in DelayBased or IncreaseUsingPadding state, got {:?}",
            state
        );
    }

    #[test]
    fn stable_loss_gradual_overuse() {
        // Test stable loss at 5% which should be ignored by the loss controller, followed by
        // a gradual increase in loss as we overuse the capacity
        let mut lbc = LossController::new();
        lbc.set_min_bitrate(Bitrate::from(50_000)); // 50 kbps
        lbc.set_max_bitrate(Bitrate::from(1_000_000_000)); // 1 Gbps

        let acknowledged_bitrate = Bitrate::from(1_000_000); // 1 Mbps
        lbc.set_acknowledged_bitrate(acknowledged_bitrate);
        lbc.set_bandwidth_estimate(Bitrate::from(1_250_000)); // 1.25Mbps

        let mut pkt_builder = PacketBuilder::new(Instant::now())
            .with_loss(0.05)
            .num_packets(26);

        // It takes a while for the maximum likelihood estimation to react to the inherent loss
        // this is why we need quite a few observations before the estimate increases to the delay
        // based bound
        // 40 observations(10 seconds) at 1Mbps
        for _ in 0..40 {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));

            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        // Gradual increase
        for inc in 0..10 {
            let loss = 0.05 + (inc as f64 / 10.0);
            pkt_builder = pkt_builder.with_loss(loss);
            lbc.set_acknowledged_bitrate(acknowledged_bitrate * (1.0 - loss));

            for _ in 0..4 {
                let result = pkt_builder.build_packets();
                lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));

                pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
            }
        }

        let LossBasedBweResult {
            bandwidth_estimate,
            state,
        } = lbc.loss_based_result();

        let estimate = bandwidth_estimate.expect("Should have an estimate");
        assert!(
            estimate < Bitrate::bps(1_000_000),
            "A gradual overuse should result in a lowered estimate"
        );
        assert_eq!(state, LossControllerState::Decreasing);
    }

    #[test]
    fn test_loss_limited_window() {
        let mut lbc = LossController::new();
        lbc.set_min_bitrate(Bitrate::kbps(50));
        lbc.set_max_bitrate(Bitrate::gbps(1));

        let acknowledged_bitrate = Bitrate::mbps(1); // 1 Mbps
        lbc.set_acknowledged_bitrate(acknowledged_bitrate);
        lbc.set_bandwidth_estimate(Bitrate::kbps(1_250)); // 1.25Mbps

        let mut pkt_builder = PacketBuilder::new(Instant::now()).num_packets(25);

        {
            // Initial observation with no loss
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));
        }

        let loss_limited = {
            // loss spike observation at 50%
            pkt_builder = pkt_builder.with_loss(0.5);
            // The receiver only gets half of the 1 Mbps.
            lbc.set_acknowledged_bitrate(Bitrate::kbps(500));
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));

            let LossBasedBweResult {
                bandwidth_estimate,
                state,
            } = lbc.loss_based_result();

            let estimate = bandwidth_estimate.expect("Should have an estimate");
            assert!(
                estimate < Bitrate::kbps(600),
                "A loss spike should've caused a significant drop in estimate, got {}",
                estimate
            );
            assert_eq!(state, LossControllerState::Decreasing);

            estimate
        };

        {
            // Recovery observation at 0% loss
            pkt_builder = pkt_builder.with_loss(0.0);
            let result = pkt_builder.build_packets();
            // Lower acknowledged bitrate to simulate reacting to estimate due to spike
            lbc.set_acknowledged_bitrate(Bitrate::kbps(300));
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));
            pkt_builder = pkt_builder.forward_time(Duration::from_millis(250));

            let LossBasedBweResult {
                bandwidth_estimate,
                state,
            } = lbc.loss_based_result();

            let estimate = bandwidth_estimate.expect("Should have an estimate");
            assert!(
                estimate >= loss_limited && estimate <= Bitrate::mbps(1),
                "During the recovery window after a loss spike the estimate should not decrease \
                    (it is held at the HOLD rate), and be bounded. loss_limited={}, estimate={}, \
                    expected <= 1 Mbps",
                loss_limited,
                estimate
            );
            assert_eq!(state, LossControllerState::Decreasing);
        }

        {
            // Another recovery observation at 0% loss, outside of the limit window
            pkt_builder = pkt_builder.num_packets(80);
            let result = pkt_builder.build_packets();
            lbc.set_acknowledged_bitrate(Bitrate::mbps(1));
            lbc.update_bandwidth_estimate(&result, Bitrate::bps(1_500_000));

            let LossBasedBweResult {
                bandwidth_estimate,
                state,
            } = lbc.loss_based_result();

            let estimate = bandwidth_estimate.expect("Should have an estimate");
            assert!(
                estimate == Bitrate::bps(1_000_000),
                "Eventually the estimate should recover but still remain bounded until the \
                    average loss caused by spike ages out"
            );
            // Up from the HOLD rate, still loss limited.
            assert_eq!(state, LossControllerState::IncreaseUsingPadding);
        }
    }

    /// One phase of a simulated send: `send_kbps` in 50 ms feedback batches of
    /// 1200 byte packets, a `loss` fraction of them lost, with the acked rate,
    /// delay-based estimate and ALR state fed before every batch.
    struct Phase {
        secs: f64,
        send_kbps: f64,
        loss: f64,
        acked_kbps: u64,
        delay_kbps: u64,
        alr: bool,
    }

    /// Drive `lbc` through `phases` in the order `SendSideBandwidthEstimator`
    /// feeds it, calling `check(t, acked, result)` after every batch.
    fn simulate(
        lbc: &mut LossController,
        phases: &[Phase],
        mut check: impl FnMut(f64, Bitrate, &LossBasedBweResult),
    ) {
        let start = Instant::now();
        let batch = Duration::from_millis(50);
        let size = DataSize::bytes(1200);
        let mut t = Duration::ZERO;
        let (mut packets_due, mut losses_due) = (0.0, 0.0);
        for phase in phases {
            let end = t + Duration::from_secs_f64(phase.secs);
            while t < end {
                packets_due +=
                    phase.send_kbps * 1000.0 * batch.as_secs_f64() / (size.as_bytes_f64() * 8.0);
                let n = packets_due as u32;
                packets_due -= n as f64;
                let packets: Vec<_> = (0..n)
                    .map(|i| {
                        losses_due += phase.loss;
                        let lost = losses_due >= 1.0;
                        if lost {
                            losses_due -= 1.0;
                        }
                        PacketResult {
                            local_send_time: start + t + batch * i / n,
                            size,
                            lost,
                        }
                    })
                    .collect();
                let acked = Bitrate::kbps(phase.acked_kbps);
                let delay_based = Bitrate::kbps(phase.delay_kbps);
                lbc.set_alr_start_time(phase.alr.then_some(start));
                lbc.set_acknowledged_bitrate(acked);
                lbc.update_bandwidth_estimate(&packets, delay_based);
                // As SendSideBandwidthEstimator::update: the delay-based
                // estimate caps the loss-based one.
                if let Some(estimate) = lbc.loss_based_result().bandwidth_estimate {
                    if estimate > delay_based {
                        lbc.set_bandwidth_estimate(delay_based);
                    }
                }
                t += batch;
                check(t.as_secs_f64(), acked, &lbc.loss_based_result());
            }
        }
    }

    #[test]
    fn estimate_not_lower_than_acked_rate_on_rate_limited_path() {
        // An application limited sender at 14 Mbps whose path drops to 8 Mbps:
        // 43 % loss for 1.5 s while the delay-based estimate falls and the
        // receiver still gets 7-9 Mbps, then the sender backs off to 3 Mbps
        // without loss.
        // (secs, send kbps, loss, acked kbps, delay-based kbps, ALR)
        let phases = [
            (10.0, 14_000.0, 0.0, 14_000, 60_000, true),
            (0.5, 14_000.0, 0.43, 9_100, 9_400, false),
            (1.0, 14_000.0, 0.43, 7_400, 7_470, false),
            (5.0, 3_000.0, 0.0, 3_000, 7_470, false),
        ]
        .map(
            |(secs, send_kbps, loss, acked_kbps, delay_kbps, alr)| Phase {
                secs,
                send_kbps,
                loss,
                acked_kbps,
                delay_kbps,
                alr,
            },
        );
        let mut lbc = LossController::new();
        lbc.set_bandwidth_estimate(Bitrate::mbps(60));
        let mut loss_limited = false;
        simulate(&mut lbc, &phases, |t, acked, result| {
            if result.state == LossControllerState::DelayBased {
                return;
            }
            loss_limited = true;
            let estimate = result.bandwidth_estimate.expect("an estimate");
            assert!(
                estimate >= acked,
                "at {t:.2} s the loss-based estimate {estimate} is below the acked rate {acked}"
            );
        });
        assert!(loss_limited, "the loss should limit the estimate");
    }

    /// A closed loop of an application limited sender, its pacer and a link: media at
    /// `MEDIA_SHARE` of the estimate, padded up to the pacer's padding rate, on a link of
    /// `capacity_kbps(t)` that drops what is sent above it. The acked rate is what got
    /// through in the last 150 ms. Calls `check(t, acked, result)` after every 50 ms batch.
    fn simulate_paced(
        lbc: &mut LossController,
        secs: f64,
        delay_kbps: u64,
        capacity_kbps: impl Fn(f64) -> f64,
        mut check: impl FnMut(f64, Bitrate, &LossBasedBweResult),
    ) {
        const MEDIA_SHARE: f64 = 0.8;
        let pacer = PacerControl::new();
        let start = Instant::now();
        let batch = Duration::from_millis(50);
        let size = DataSize::bytes(1200);
        let delay_based = Bitrate::kbps(delay_kbps);
        let mut t = Duration::ZERO;
        let (mut packets_due, mut losses_due) = (0.0, 0.0);
        let mut delivered_kbps = [0.0; 3];
        while t.as_secs_f64() < secs {
            let result = lbc.loss_based_result();
            let estimate = result.bandwidth_estimate.expect("an estimate");
            let padding = loss_recovery_padding_rate(result.state, estimate, Bitrate::INFINITY);
            let padding_rate = pacer
                .calculate(true, estimate, false, Bitrate::ZERO, padding)
                .padding_rate;
            let send_kbps = (estimate.as_f64() * MEDIA_SHARE).max(padding_rate.as_f64()) / 1000.0;
            let capacity = capacity_kbps(t.as_secs_f64());
            let loss = ((send_kbps - capacity) / send_kbps).max(0.0);

            packets_due += send_kbps * 1000.0 * batch.as_secs_f64() / (size.as_bytes_f64() * 8.0);
            let n = packets_due as u32;
            packets_due -= n as f64;
            let packets: Vec<_> = (0..n)
                .map(|i| {
                    losses_due += loss;
                    let lost = losses_due >= 1.0;
                    if lost {
                        losses_due -= 1.0;
                    }
                    PacketResult {
                        local_send_time: start + t + batch * i / n,
                        size,
                        lost,
                    }
                })
                .collect();
            delivered_kbps.rotate_left(1);
            delivered_kbps[2] = send_kbps * (1.0 - loss);
            let acked = Bitrate::kbps((delivered_kbps.iter().sum::<f64>() / 3.0) as u64);

            lbc.set_alr_start_time(None);
            lbc.set_acknowledged_bitrate(acked);
            lbc.update_bandwidth_estimate(&packets, delay_based);
            if let Some(estimate) = lbc.loss_based_result().bandwidth_estimate {
                if estimate > delay_based {
                    lbc.set_bandwidth_estimate(delay_based);
                }
            }
            t += batch;
            check(t.as_secs_f64(), acked, &lbc.loss_based_result());
        }
    }

    /// The run of `padding_lifts_app_limited_estimate_after_loss`.
    #[derive(Debug)]
    struct Recovery {
        /// The lowest estimate after the loss started.
        lowest: Bitrate,
        /// When the estimate first increased while loss limited.
        increasing_at: Option<f64>,
        /// When the estimate reached 6 Mbps after the loss.
        recovered_at: Option<f64>,
        /// Whether the pacer padded above its normal target.
        padded: bool,
        /// Estimate, acked rate and state at the end.
        end: (Bitrate, Bitrate, LossControllerState),
    }

    fn recover_after_loss(padding_duration: Duration) -> Recovery {
        // An application limited sender at 0.8 x the estimate on a 8 Mbps link that
        // drops to 400 kbps for 1 s.
        let mut lbc = LossController::new();
        lbc.config.padding_duration = padding_duration;
        lbc.set_bandwidth_estimate(Bitrate::mbps(4));
        let capacity = |t: f64| {
            if (5.0..6.0).contains(&t) {
                400.0
            } else {
                8_000.0
            }
        };
        let mut run = Recovery {
            lowest: Bitrate::INFINITY,
            increasing_at: None,
            recovered_at: None,
            padded: false,
            end: (
                Bitrate::ZERO,
                Bitrate::ZERO,
                LossControllerState::DelayBased,
            ),
        };
        simulate_paced(&mut lbc, 16.0, 8_000, capacity, |t, acked, result| {
            let estimate = result.bandwidth_estimate.unwrap();
            run.padded |= result.state == LossControllerState::IncreaseUsingPadding;
            if t > 5.0 {
                run.lowest = run.lowest.min(estimate);
                if run.increasing_at.is_none()
                    && matches!(
                        result.state,
                        LossControllerState::Increasing | LossControllerState::IncreaseUsingPadding
                    )
                {
                    run.increasing_at = Some(t);
                }
            }
            if t > 6.0 && run.recovered_at.is_none() && estimate >= Bitrate::kbps(6_000) {
                run.recovered_at = Some(t);
            }
            run.end = (estimate, acked, result.state);
        });
        run
    }

    #[test]
    fn padding_lifts_app_limited_estimate_after_loss() {
        // After the loss the media alone cannot raise the acked rate, which bounds the
        // increase while loss limited; padding up to the estimate can.
        let padded = recover_after_loss(Duration::from_secs(2));
        let unpadded = recover_after_loss(Duration::ZERO);
        println!("padded {padded:?}\nunpadded {unpadded:?}");

        assert!(
            unpadded.lowest < Bitrate::kbps(1_000),
            "the loss should cut the estimate"
        );
        assert!(!unpadded.padded);
        let (estimate, acked, state) = unpadded.end;
        assert!(
            unpadded.recovered_at.is_none()
                && state != LossControllerState::DelayBased
                && estimate <= acked * 1.5,
            "without padding the estimate stays near the acked rate"
        );

        assert!(
            padded.lowest < Bitrate::kbps(1_000),
            "the loss should cut the estimate"
        );
        assert!(padded.padded);
        let increasing_at = padded.increasing_at.expect("a loss limited increase");
        let recovered_at = padded
            .recovered_at
            .expect("the estimate should recover to 6 Mbps");
        assert!(
            recovered_at - increasing_at < 4.0,
            "6 Mbps reached {:.2} s after the increase started",
            recovered_at - increasing_at
        );
        assert_eq!(padded.end.2, LossControllerState::DelayBased);
    }

    /// libWebRTC's DecreaseAfterPadding: 50 % loss until the state is `Decreasing`,
    /// then no loss at an acked rate that does not follow the estimate. Returns the
    /// state and estimate after every observation, with the time since the loss.
    fn loss_then_unacked_padding(
        lbc: &mut LossController,
    ) -> Vec<(Duration, LossControllerState, Bitrate)> {
        let step = Duration::from_millis(300);
        let mut elapsed = Duration::ZERO;
        lbc.set_bandwidth_estimate(Bitrate::kbps(2_500));
        lbc.set_acknowledged_bitrate(Bitrate::kbps(1_250));
        let mut pkt_builder = PacketBuilder::new(Instant::now())
            .num_packets(25)
            .with_loss(0.5);
        while lbc.loss_based_result().state != LossControllerState::Decreasing {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::INFINITY);
            pkt_builder.now += step;
            elapsed += step;
            assert!(elapsed < Duration::from_secs(10), "no decrease");
        }
        lbc.set_acknowledged_bitrate(Bitrate::kbps(600));
        pkt_builder = pkt_builder.with_loss(0.0);
        elapsed = Duration::ZERO;
        let mut run = vec![];
        for _ in 0..40 {
            let result = pkt_builder.build_packets();
            lbc.update_bandwidth_estimate(&result, Bitrate::INFINITY);
            pkt_builder.now += step;
            elapsed += step;
            let result = lbc.loss_based_result();
            run.push((elapsed, result.state, result.bandwidth_estimate.unwrap()));
        }
        run
    }

    #[test]
    fn increase_using_padding_ends_after_padding_duration() {
        // The padding is not acked, so the estimate stops at the bound by the acked rate
        // and the state ends a padding duration after the last new high.
        let mut lbc = LossController::new();
        let run = loss_then_unacked_padding(&mut lbc);
        let padding = |(_, s, _): &&(Duration, LossControllerState, Bitrate)| {
            *s == LossControllerState::IncreaseUsingPadding
        };
        let first = run.iter().position(|r| padding(&r)).expect("padding");
        let end = first
            + run[first..]
                .iter()
                .position(|r| !padding(&r))
                .expect("an end");
        let (left, state, estimate) = run[end];
        let (_, _, highest) = run[end - 1];
        let reached = run[first..end].iter().find(|r| r.2 == highest).unwrap().0;
        println!(
            "padding from {:?}, highest {highest} at {reached:?}, {state:?} at {left:?}",
            run[first].0
        );
        assert_eq!(state, LossControllerState::Decreasing);
        assert_eq!(estimate, highest, "HOLD at the padded estimate");
        // The 600 kbps acked rate x 1.2 in the HOLD region.
        assert_eq!(highest, Bitrate::kbps(720));
        let held = left - reached;
        // Observations are 300 ms apart.
        assert!(
            held > Duration::from_secs(2) && held <= Duration::from_millis(2_300),
            "left {held:?} after the highest estimate"
        );
    }

    #[test]
    fn increasing_without_padding_duration() {
        // A zero padding duration (libWebRTC's `PaddingDuration:0`) increases without
        // padding.
        let mut lbc = LossController::new();
        lbc.config.padding_duration = Duration::ZERO;
        let run = loss_then_unacked_padding(&mut lbc);
        assert!(run.iter().any(|r| r.1 == LossControllerState::Increasing));
        assert!(
            !run.iter()
                .any(|r| r.1 == LossControllerState::IncreaseUsingPadding)
        );
    }

    struct PacketBuilder {
        now: Instant,
        rng: Rng,
        loss_rate: f64,
        send_distribution: LogNormalDistribution,
        recv_distribution: LogNormalDistribution,
        num_packets: u32,
        packet_size: DataSize,
    }

    impl PacketBuilder {
        fn new(now: Instant) -> Self {
            Self {
                now,
                rng: Rng::with_seed(34791910),
                loss_rate: 0.0,
                send_distribution: LogNormalDistribution {
                    mean: 0.05,
                    std_dev: 1.0,
                },
                recv_distribution: LogNormalDistribution {
                    mean: 4.0,
                    std_dev: 10.0,
                },
                num_packets: 10,
                packet_size: DataSize::bytes(1200),
            }
        }

        fn forward_time(mut self, by: Duration) -> Self {
            self.now += by;
            self
        }

        fn with_loss(mut self, loss_rate: f64) -> Self {
            self.loss_rate = loss_rate;
            self
        }

        fn num_packets(mut self, packets: u32) -> Self {
            self.num_packets = packets;
            self
        }

        fn build_packets(&mut self) -> Vec<PacketResult> {
            let mut last_send_time = self.now;
            let mut last_recv_time = self.now;
            let mut result: Vec<PacketResult> = Vec::with_capacity(self.num_packets as usize);

            for _ in 0..self.num_packets {
                let lost = self.rng.f64() <= self.loss_rate;
                let first_send_time = last_send_time
                    + Duration::from_secs_f64(
                        self.send_distribution.sample(&mut self.rng) / 1000.0,
                    );
                let recv_time = last_recv_time
                    + Duration::from_secs_f64(
                        self.recv_distribution.sample(&mut self.rng) / 1000.0,
                    );

                result.push(PacketResult {
                    local_send_time: first_send_time,
                    size: self.packet_size,
                    lost,
                });

                last_send_time = first_send_time;
                if !lost {
                    last_recv_time = recv_time;
                }
            }

            result
        }
    }

    struct LogNormalDistribution {
        mean: f64,
        std_dev: f64,
    }

    impl LogNormalDistribution {
        fn sample(&self, rng: &mut Rng) -> f64 {
            let normal = normal_distribution(rng);
            let location =
                (self.mean.powi(2) / (self.mean.powi(2) + self.std_dev.powi(2)).sqrt()).ln();
            let scale = (1.0 + (self.std_dev / self.mean).powi(2)).ln().sqrt();

            (location + scale * normal).exp()
        }
    }

    fn normal_distribution(rng: &mut Rng) -> f64 {
        let u1 = rng.f64();
        let u2 = rng.f64();

        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
