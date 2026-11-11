//! Google Congestion Control (GoogCC) Bandwidth Estimation based on TWCC feedback.
//!
//! This implementation is ported from libWebRTC's GoogCC and goes beyond the simplified
//! IETF draft (<https://datatracker.ietf.org/doc/html/draft-ietf-rmcat-gcc-02>) to include
//! WebRTC's production features:
//!
//! - Delay-based control (trendline estimator with AIMD rate control)
//! - Loss-based control (with inherent loss rate estimation)
//! - Probe controller with state machine and multi-stage probing strategy
//! - ALR (Application Limited Region) detection and periodic probing
//! - Link capacity estimation from ALR probes
//!
//! The probe controller in particular closely matches WebRTC's `ProbeController` behavior
//! and default constants, enabling compatible bandwidth discovery with WebRTC endpoints.

use std::cmp::Ordering;
use std::fmt;
use std::time::{Duration, Instant};

use crate::Reason;
use crate::rtp_::{Bitrate, DataSize, TwccClusterId, TwccSendRecord, TwccSeq};
use crate::util::Soonest;

mod acked_bitrate_estimator;
mod alr_detector;
pub(crate) mod api;
mod delay;
mod link_capacity_estimator;
mod loss_controller;
mod macros;
mod probe;
mod smoother;
mod time;

use acked_bitrate_estimator::AckedBitrateEstimator;
use alr_detector::AlrDetector;
use delay::DelayController;
use link_capacity_estimator::LinkCapacityEstimator;
use loss_controller::{LossController, LossControllerState};
use macros::log_loss;
use smoother::EstimateSmoother;

pub(crate) use macros::{log_pacer_media_debt, log_pacer_padding_debt};
#[cfg(test)]
pub(crate) use probe::ProbeKind;
pub(crate) use probe::{BandwidthLimitedCause, ProbeEstimator};
pub(crate) use probe::{ProbeClusterState, ProbeControl};

#[cfg(feature = "_internal_test_exports")]
pub use probe::ProbeClusterConfig;
#[cfg(not(feature = "_internal_test_exports"))]
pub(crate) use probe::ProbeClusterConfig;

const INITIAL_BITRATE_WINDOW: Duration = Duration::from_millis(500);
const BITRATE_WINDOW: Duration = Duration::from_millis(150);
const STARTUP_PHASE: Duration = Duration::from_secs(2);

pub struct Bwe {
    bwe: SendSideBandwidthEstimator,
    desired_bitrate: Bitrate,
    minimum_bitrate: Bitrate,
    smoother: EstimateSmoother,
}

impl Bwe {
    pub fn new(initial: Bitrate) -> Self {
        let send_side_bwe = SendSideBandwidthEstimator::new(initial);
        Bwe {
            bwe: send_side_bwe,
            desired_bitrate: Bitrate::ZERO,
            minimum_bitrate: Bitrate::ZERO,
            smoother: EstimateSmoother::new(),
        }
    }

    pub fn handle_timeout(&mut self, now: Instant, do_probe: bool) -> Option<ProbeClusterConfig> {
        let result = self.bwe.handle_timeout(self.desired_bitrate, do_probe, now);
        if let Some(estimate) = self.bwe.last_estimate() {
            self.smoother.record(now, estimate);
        }
        result
    }

    pub fn start_probe(&mut self, config: ProbeClusterConfig, now: Instant) -> bool {
        self.bwe.start_probe(config, now)
    }

    pub fn end_probe(&mut self, now: Instant, cluster_id: TwccClusterId) {
        self.bwe.end_probe(now, cluster_id);
    }

    pub fn reset(&mut self, init_bitrate: Bitrate) {
        self.bwe.reset(init_bitrate);
    }

    pub fn update<'t>(
        &mut self,
        records: impl Iterator<Item = &'t crate::rtp_::TwccSendRecord>,
        now: Instant,
    ) {
        self.bwe.update(records, now);
    }

    pub fn poll_estimate(&mut self, smoothing: bool) -> Option<Bitrate> {
        self.smoother.poll(smoothing)
    }

    pub fn poll_timeout(&self) -> (Option<Instant>, Reason) {
        self.bwe.poll_timeout()
    }

    pub fn last_estimate(&self) -> Option<Bitrate> {
        self.bwe.last_estimate()
    }

    pub fn on_media_sent(&mut self, payload_size: DataSize, is_padding: bool, now: Instant) {
        if !is_padding {
            // Update ALR detector with media bytes sent
            self.bwe.on_media_sent(payload_size, now);
        }
    }

    pub fn is_overusing(&self) -> bool {
        self.bwe.is_overusing()
    }

    pub fn set_bitrate(&mut self, minimum: Bitrate, desired: Bitrate) {
        self.minimum_bitrate = minimum.min(desired);
        self.desired_bitrate = desired;
    }

    pub fn minimum_bitrate(&self) -> Bitrate {
        self.minimum_bitrate
    }
}

struct SendSideBandwidthEstimator {
    delay_controller: DelayController,
    loss_controller: LossController,
    acked_bitrate_estimator: AckedBitrateEstimator,
    probe_control: ProbeControl,
    probe_estimator: ProbeEstimator,
    // First nonempty transport feedback; starts loss estimation and enables ALR probing.
    started_at: Option<Instant>,
    alr_detector: AlrDetector,
    link_capacity_estimator: LinkCapacityEstimator,
    last_updated_estimate: Option<(Bitrate, BandwidthLimitedCause)>,
}

impl SendSideBandwidthEstimator {
    pub fn new(initial_bitrate: Bitrate) -> Self {
        let mut alr_detector = AlrDetector::new();
        alr_detector.set_estimated_bitrate(initial_bitrate);

        let mut loss_controller = LossController::new();
        loss_controller.set_bandwidth_estimate(initial_bitrate);

        Self {
            delay_controller: DelayController::new(initial_bitrate),
            loss_controller,
            acked_bitrate_estimator: AckedBitrateEstimator::new(
                INITIAL_BITRATE_WINDOW,
                BITRATE_WINDOW,
            ),
            probe_control: ProbeControl::new(),
            probe_estimator: ProbeEstimator::new(),
            started_at: None,
            alr_detector,
            link_capacity_estimator: LinkCapacityEstimator::new(),
            last_updated_estimate: None,
        }
    }

    /// Whether the delay-based detector currently signals overuse.
    ///
    /// This is useful for gating behaviors (like padding/probing) that would otherwise
    /// re-excite the system while we're already congested.
    pub fn is_overusing(&self) -> bool {
        self.delay_controller.is_overusing()
    }

    /// Update ALR detector with actual bytes sent.
    ///
    /// Should be called for media packets (not padding/probes).
    /// This is typically called from the session's packet sending logic.
    pub fn on_media_sent(&mut self, bytes: DataSize, now: Instant) {
        self.alr_detector.on_bytes_sent(bytes, now);
    }

    /// Record a packet from a TWCC report.
    pub fn update<'t>(&mut self, records: impl Iterator<Item = &'t TwccSendRecord>, now: Instant) {
        let send_records: Vec<_> = records.collect();
        if send_records.is_empty() {
            return;
        }
        let _ = self.started_at.get_or_insert(now);

        // Feed records to probe estimator for analysis and process any new probe results
        let mut latest_probe_result = None;
        for (config, bitrate) in self.probe_estimator.update(send_records.iter().copied()) {
            latest_probe_result = Some(bitrate);

            // Update link capacity estimator for every successful ALR probe, not just the latest.
            // The estimator internally takes the max of all probe results, building up knowledge
            // of proven link capacity. This differs from the delay controller, which only receives
            // the latest probe result (matching WebRTC's FetchAndResetLastEstimatedBitrate behavior).
            if config.is_alr_probe() {
                self.link_capacity_estimator.update_from_probe(bitrate, now);
            }
        }

        let mut acked_packets = vec![];

        let mut max_rtt = None;
        let mut count = 0;
        let mut lost = 0;
        for record in send_records.iter() {
            count += 1;
            let Ok(acked_packet) = (*record).try_into() else {
                lost += 1;
                continue;
            };
            acked_packets.push(acked_packet);
            max_rtt = max_rtt.max(record.rtt());
        }
        acked_packets.sort_by(AckedPacket::order_by_receive_time);

        for acked_packet in acked_packets.iter() {
            self.acked_bitrate_estimator
                .update(acked_packet.remote_recv_time, acked_packet.size);
        }

        let acked_bitrate = self.acked_bitrate_estimator.current_estimate();

        // Use the latest probe result from this update, if any
        let probe_result = latest_probe_result;

        let is_probe_result = probe_result.is_some();

        // Update delay controller with the latest probe result
        let maybe_estimate =
            self.delay_controller
                .update(&acked_packets, acked_bitrate, probe_result, now);

        let Some(delay_estimate) = maybe_estimate else {
            return;
        };

        let loss = if count == 0 {
            0.0
        } else {
            lost as f64 / count as f64
        };
        log_loss!(loss);

        // During startup with no loss, use delay-based estimate directly
        if in_startup_phase(self.started_at, now) && loss <= 0.001 {
            self.loss_controller.set_bandwidth_estimate(delay_estimate);
            return;
        }

        // When probe succeeds, set bandwidth directly
        if is_probe_result {
            self.loss_controller.set_bandwidth_estimate(delay_estimate);
        }

        if let Some(acked_bitrate) = acked_bitrate {
            self.loss_controller.set_acknowledged_bitrate(acked_bitrate);
        }

        // This corresponds to UpdateLossBasedEstimator + UpdateEstimate
        self.loss_controller
            .update_bandwidth_estimate(&send_records, delay_estimate);

        // Loss-based result is capped by delay_based_limit
        let loss_result = self.loss_controller.loss_based_result();
        if let Some(loss_estimate) = loss_result.bandwidth_estimate {
            if loss_estimate > delay_estimate {
                // Loss controller produced higher estimate than delay controller
                // Cap it at delay estimate (delay controller is the upper limit)
                self.loss_controller.set_bandwidth_estimate(delay_estimate);
            }
        }

        // Feed the (possibly combined) estimate into subcomponents wanting it.
        self.propagate_estimate();
    }

    pub fn poll_timeout(&self) -> (Option<Instant>, Reason) {
        let delay_timeout = Some(self.delay_controller.poll_timeout());
        let probe_timeout = Some(self.probe_control.poll_timeout());
        let probe_estimator_timeout = Some(self.probe_estimator.poll_timeout());
        (delay_timeout, Reason::BweDelayControl)
            .soonest((probe_timeout, Reason::BweProbeControl))
            .soonest((probe_estimator_timeout, Reason::BweProbeEstimator))
    }

    /// Handle periodic timeout for BWE components.
    pub fn handle_timeout(
        &mut self,
        desired_bitrate: Bitrate,
        do_probe: bool,
        now: Instant,
    ) -> Option<ProbeClusterConfig> {
        self.delay_controller
            .handle_timeout(self.acked_bitrate_estimator.current_estimate(), now);

        // Update probe control with desired bitrate.
        self.probe_control.set_desired_bitrate(desired_bitrate);

        // Empty media queues are application-limited too. Advance the budget
        // without charging probe/padding bytes, including before the first media.
        if do_probe {
            self.alr_detector.handle_timeout(now);
        }

        // Get ALR state and forward to both probe control and loss controller
        let alr_start_time = self.alr_detector.alr_start_time();
        // ALR probing requires feedback for at least one sent packet.
        if let Some(t) = alr_start_time.filter(|_| self.started_at.is_some()) {
            self.probe_control.set_alr_start_time(t);
        } else {
            self.probe_control.set_alr_stop_time(now);
        }

        self.loss_controller.set_alr_start_time(alr_start_time);

        // Get link capacity estimate and forward to loss controller.
        let link_capacity = self.link_capacity_estimator.capacity_estimate(now);
        self.loss_controller
            .set_link_capacity_estimate(link_capacity);

        // Clean up expired probe cluster state
        self.probe_estimator.handle_timeout(now);

        // Feed the current estimate into subcontrollers, if it changed.
        self.propagate_estimate();

        // If we can't probe, clear any pending/active probes
        if !do_probe {
            self.probe_estimator.clear_probes();
        }

        self.probe_control.enable(do_probe);

        // Timeout-driven probe logic (WebRTC `Process()` equivalent).
        self.probe_control.handle_timeout(now)
    }

    fn propagate_estimate(&mut self) {
        // Do we have a value?
        let Some(estimate) = self.last_estimate() else {
            return;
        };
        // Did it, or the cause limiting it, change? The cause changes without the estimate
        // when an overuse ends while the estimate holds still (in AIMD hold, or capped by the
        // acked rate of an application limited sender): the probe controller must not keep
        // the overuse cause, which blocks every probe.
        let cause = self.bandwidth_limited_cause();
        if self.last_updated_estimate == Some((estimate, cause)) {
            return;
        }

        self.probe_control.set_estimated_bitrate(estimate, cause);
        self.alr_detector.set_estimated_bitrate(estimate);

        // Don't update until this changes.
        self.last_updated_estimate = Some((estimate, cause));
    }

    fn bandwidth_limited_cause(&self) -> BandwidthLimitedCause {
        if self.delay_controller.is_overusing() {
            return BandwidthLimitedCause::DelayBasedLimitedDelayIncreased;
        }

        match self.loss_controller.loss_based_result().state {
            _ if !self.is_loss_limited() => BandwidthLimitedCause::DelayBasedLimited,
            LossControllerState::DelayBased => BandwidthLimitedCause::DelayBasedLimited,
            LossControllerState::Increasing => BandwidthLimitedCause::LossLimitedBweIncreasing,
            LossControllerState::Decreasing => BandwidthLimitedCause::LossLimitedBwe,
        }
    }

    /// Whether the loss-based estimate limits the estimate: the loss controller is out of
    /// `DelayBased` and its estimate is below the delay-based one.
    ///
    /// libWebRTC's `LossBasedBweV2` leaves `kDelayBasedEstimate` only when its bounded
    /// estimate is below the delay-based estimate, but decides that once per observation
    /// (250 ms or more) while the delay-based estimate changes with every feedback. Here the
    /// comparison is made with the current delay-based estimate. Otherwise, once an overuse
    /// has cut the delay-based estimate below the loss-based one (which is never below the
    /// acked rate), the state from the last observation reports a delay-based limit as a
    /// loss limit, and keeps probes blocked, until the next observation. libWebRTC reports
    /// the overuse itself as the cause (as `bandwidth_limited_cause` does), but not the
    /// time between its end and the next observation.
    fn is_loss_limited(&self) -> bool {
        let result = self.loss_controller.loss_based_result();
        if result.state == LossControllerState::DelayBased {
            return false;
        }
        let delay_based = self.delay_controller.last_estimate();
        result
            .bandwidth_estimate
            .is_some_and(|loss_based| delay_based.is_none_or(|d| loss_based < d))
    }

    /// Get the latest estimate.
    pub fn last_estimate(&self) -> Option<Bitrate> {
        let delay_estimate = self.delay_controller.last_estimate();

        let loss_result = self.loss_controller.loss_based_result();

        // Only apply loss-based limiting when actively in a loss-limiting state
        match loss_result.state {
            LossControllerState::DelayBased => {
                // Loss controller defers to delay-based estimate
                delay_estimate
            }
            LossControllerState::Decreasing | LossControllerState::Increasing => {
                // Loss controller is actively limiting or recovering
                match (delay_estimate, loss_result.bandwidth_estimate) {
                    (Some(de), Some(le)) => Some(de.min(le)),
                    (None, le @ Some(_)) => le,
                    (de @ Some(_), None) => de,
                    (None, None) => None,
                }
            }
        }
    }

    /// Start analyzing a probe cluster.
    ///
    /// This should be called when the pacer starts sending a probe cluster,
    /// to tell the estimator which cluster to watch for in TWCC feedback.
    /// Returns `true` if the probe was started, `false` if rejected.
    pub fn start_probe(&mut self, config: ProbeClusterConfig, now: Instant) -> bool {
        self.probe_estimator.probe_start(config, now)
    }

    /// End a probe cluster and mark it for cleanup.
    ///
    /// This should be called when the pacer finishes sending a probe cluster.
    /// The estimator will continue collecting feedback for a cluster history period
    /// to allow late-arriving TWCC reports to refine the estimate.
    pub fn end_probe(&mut self, now: Instant, cluster_id: TwccClusterId) {
        self.probe_estimator.end_probe(now, cluster_id);
    }

    pub fn reset(&mut self, init_bitrate: Bitrate) {
        *self = Self::new(init_bitrate);
    }
}

/// A RTP packet that has been sent and acknowledged by the receiver in a TWCC report.
#[derive(Debug, Copy, Clone)]
pub struct AckedPacket {
    /// The TWCC sequence number
    seq_no: TwccSeq,
    /// The size of the packets in bytes.
    size: DataSize,
    /// When we sent the packet
    local_send_time: Instant,
    /// When the packet was received at the remote, note this Instant is only usable with other
    /// instants of the same type i.e. those that represent a TWCC reported receive time for this
    /// session.
    remote_recv_time: Instant,
    /// The local time when received confirmation that the other side received the seq i.e. when we
    /// received the TWCC report for this packet.
    local_recv_time: Instant,
}

impl AckedPacket {
    fn rtt(&self) -> Duration {
        self.local_recv_time - self.local_send_time
    }

    fn order_by_receive_time(lhs: &Self, rhs: &Self) -> Ordering {
        if lhs.remote_recv_time != rhs.remote_recv_time {
            lhs.remote_recv_time.cmp(&rhs.remote_recv_time)
        } else if lhs.local_send_time != rhs.local_send_time {
            lhs.local_send_time.cmp(&rhs.local_send_time)
        } else {
            lhs.seq_no.cmp(&rhs.seq_no)
        }
    }
}

// NB: Extracted for lifetime reasons
fn in_startup_phase(started_at: Option<Instant>, now: Instant) -> bool {
    started_at
        .map(|s| now.duration_since(s) <= STARTUP_PHASE)
        .unwrap_or(false)
}

impl TryFrom<&TwccSendRecord> for AckedPacket {
    type Error = ();

    fn try_from(value: &TwccSendRecord) -> Result<Self, Self::Error> {
        let Some(remote_recv_time) = value.remote_recv_time() else {
            return Err(());
        };
        let Some(local_recv_time) = value.local_recv_time() else {
            return Err(());
        };

        Ok(Self {
            seq_no: value.seq(),
            size: value.size().into(),
            local_send_time: value.local_send_time(),
            remote_recv_time,
            local_recv_time,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandwidthUsage {
    Overuse,
    Normal,
    Underuse,
}

impl fmt::Display for BandwidthUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandwidthUsage::Overuse => write!(f, "overuse"),
            BandwidthUsage::Normal => write!(f, "normal"),
            BandwidthUsage::Underuse => write!(f, "underuse"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rtp_::TwccPacketId;

    #[test]
    fn alr_probing_waits_for_nonempty_transport_feedback() {
        let start = Instant::now();
        let desired = Bitrate::mbps(3);
        let mut bwe = SendSideBandwidthEstimator::new(Bitrate::kbps(300));

        // Initial discovery must work before feedback or media exists.
        assert!(bwe.handle_timeout(desired, true, start).is_some());
        assert!(bwe.handle_timeout(desired, true, start).is_some());
        for second in 1..=7 {
            let now = start + Duration::from_secs(second);
            bwe.update(std::iter::empty(), now);
            assert!(
                bwe.handle_timeout(desired, true, now).is_none(),
                "empty feedback must not authorize periodic ALR probes"
            );
        }
        assert!(bwe.alr_detector.alr_start_time().is_some());
        assert!(bwe.started_at.is_none());

        // Feedback for an actual sent packet authorizes ALR discovery.
        let now = start + Duration::from_secs(8);
        let record = TwccSendRecord::test_new(
            TwccPacketId::new(1u64),
            now - Duration::from_millis(20),
            240,
            now,
            Some(now - Duration::from_millis(10)),
        );
        bwe.update(std::iter::once(&record), now);
        let probe = bwe.handle_timeout(desired, true, now).unwrap();
        assert!(probe.is_alr_probe());

        // A reset starts discovery afresh, including the feedback requirement.
        bwe.reset(Bitrate::kbps(300));
        assert!(bwe.handle_timeout(desired, true, now).is_some());
        assert!(bwe.handle_timeout(desired, true, now).is_some());
        for second in 1..=7 {
            assert!(
                bwe.handle_timeout(desired, true, now + Duration::from_secs(second))
                    .is_none()
            );
        }
    }

    /// A sender of 1200 byte packets over a link with a queue and 20 ms of propagation
    /// delay, feeding the estimator as `Session` does: TWCC reports every 50 ms, each
    /// followed by a timeout.
    struct Link {
        start: Instant,
        /// Time since `start`.
        t: Duration,
        /// When the link's queue empties.
        queue_free: Instant,
        seq: u64,
        /// Sent packets not yet reported: sequence number, send and receive time.
        in_flight: Vec<(u64, Instant, Instant)>,
        next_report: Instant,
    }

    impl Link {
        fn new() -> Self {
            let start = Instant::now();
            Link {
                start,
                t: Duration::ZERO,
                queue_free: start,
                seq: 0,
                in_flight: vec![],
                next_report: start,
            }
        }

        /// Send at `send` for `secs` over a link of `link`.
        fn run(
            &mut self,
            bwe: &mut SendSideBandwidthEstimator,
            secs: f64,
            send: Bitrate,
            link: Bitrate,
        ) {
            let size = DataSize::bytes(1200);
            let end = self.t + Duration::from_secs_f64(secs);
            while self.t < end {
                let sent = self.start + self.t;
                let received = self.queue_free.max(sent) + size / link;
                self.queue_free = received;
                self.seq += 1;
                let remote_recv = received + Duration::from_millis(20);
                self.in_flight.push((self.seq, sent, remote_recv));
                self.t += size / send;
                let now = self.start + self.t;
                if now < self.next_report {
                    continue;
                }
                let arrived = self.in_flight.iter().take_while(|r| r.2 <= now).count();
                let report: Vec<_> = self
                    .in_flight
                    .drain(..arrived)
                    .map(|(seq, sent, recv)| {
                        TwccSendRecord::test_new(
                            TwccPacketId::new(seq),
                            sent,
                            1200,
                            now,
                            Some(recv),
                        )
                    })
                    .collect();
                bwe.update(report.iter(), now);
                let _ = bwe.handle_timeout(Bitrate::mbps(10), true, now);
                self.next_report = now + Duration::from_millis(50);
            }
        }
    }

    #[test]
    fn cause_reaches_probe_control_without_estimate_change() {
        // 3 Mbit/s on an 8 Mbit/s link, then the link drops to 2 Mbit/s. The overuse is
        // detected before the estimate is cut: the probe controller must know at once.
        let mut bwe = SendSideBandwidthEstimator::new(Bitrate::mbps(4));
        let mut link = Link::new();
        link.run(&mut bwe, 3.0, Bitrate::mbps(3), Bitrate::mbps(8));
        let before = bwe.last_estimate();
        let mut overused = false;
        for _ in 0..40 {
            link.run(&mut bwe, 0.05, Bitrate::mbps(3), Bitrate::mbps(2));
            if bwe.is_overusing() {
                overused = true;
                break;
            }
        }
        assert!(overused, "the 2 Mbit/s link should be detected as overuse");
        assert_eq!(bwe.last_estimate(), before, "precondition: not cut yet");
        assert_eq!(
            bwe.probe_control.last_cause(),
            BandwidthLimitedCause::DelayBasedLimitedDelayIncreased
        );

        // The sender backs off and the queue drains: once the overuse is over the cause is
        // the delay-based limit again.
        for _ in 0..40 {
            link.run(&mut bwe, 0.05, Bitrate::kbps(500), Bitrate::mbps(2));
            if !bwe.is_overusing() {
                break;
            }
        }
        assert!(!bwe.is_overusing(), "the overuse should end");
        assert_eq!(
            bwe.probe_control.last_cause(),
            BandwidthLimitedCause::DelayBasedLimited
        );
    }

    /// Packets of 1200 bytes sent at `send_kbps` from `from` for `secs`, each `lose_every`th
    /// one lost, as TWCC send records.
    fn records(
        start: Instant,
        from: Duration,
        secs: f64,
        send_kbps: u64,
        lose_every: u64,
        seq: &mut u64,
    ) -> Vec<TwccSendRecord> {
        let size = 1200;
        let interval = Duration::from_secs_f64(size as f64 * 8.0 / (send_kbps as f64 * 1000.0));
        let n = (secs / interval.as_secs_f64()) as u32;
        (0..n)
            .map(|i| {
                *seq += 1;
                let sent = start + from + interval * i;
                let recv = sent + Duration::from_millis(30);
                let lost = (*seq).is_multiple_of(lose_every);
                TwccSendRecord::test_new(
                    TwccPacketId::new(*seq),
                    sent,
                    size,
                    recv,
                    (!lost).then_some(recv),
                )
            })
            .collect()
    }

    /// An estimator whose delay controller is at `delay_estimate` and whose loss controller is
    /// out of `DelayBased`: fed 3 s of 12 Mbps with 20 % loss and 10 Mbps acked against a
    /// 20 Mbps delay-based estimate, then a 30 ms feedback, too short to complete an
    /// observation, with the delay-based estimate at `delay_estimate` (in the order
    /// `SendSideBandwidthEstimator::update` feeds it).
    fn loss_controller_out_of_delay_based(delay_estimate: Bitrate) -> SendSideBandwidthEstimator {
        let mut bwe = SendSideBandwidthEstimator::new(delay_estimate);
        let lc = &mut bwe.loss_controller;
        lc.set_bandwidth_estimate(Bitrate::mbps(20));
        let start = Instant::now();
        let mut seq = 0;
        // Feedback of 300 ms: each completes an observation (250 ms).
        let batch = Duration::from_millis(300);
        let mut t = Duration::ZERO;
        while t < Duration::from_secs(3) {
            let records = records(start, t, 0.3, 12_000, 5, &mut seq);
            lc.set_acknowledged_bitrate(Bitrate::mbps(10));
            lc.update_bandwidth_estimate(&records.iter().collect::<Vec<_>>(), Bitrate::mbps(20));
            t += batch;
        }
        let state = lc.loss_based_result().state;
        assert_ne!(state, LossControllerState::DelayBased, "precondition");
        let records = records(start, t, 0.03, 12_000, 5, &mut seq);
        lc.update_bandwidth_estimate(&records.iter().collect::<Vec<_>>(), delay_estimate);
        assert_eq!(lc.loss_based_result().state, state, "precondition");
        bwe
    }

    #[test]
    fn delay_dip_below_loss_estimate_is_not_loss_limited() {
        // A delay overuse cut the delay-based estimate to 8.5 Mbps, below the 10 Mbps acked
        // rate and so below the loss-based estimate, before the loss controller completed its
        // next observation: the delay-based estimate limits the estimate.
        let delay = Bitrate::kbps(8_554);
        let bwe = loss_controller_out_of_delay_based(delay);
        let loss = bwe.loss_controller.loss_based_result();
        println!("loss-based {loss:?}");
        assert!(loss.bandwidth_estimate.unwrap() > delay);
        assert_eq!(bwe.last_estimate(), Some(delay));
        assert!(!bwe.is_loss_limited());
        assert_eq!(
            bwe.bandwidth_limited_cause(),
            BandwidthLimitedCause::DelayBasedLimited
        );
    }

    #[test]
    fn loss_estimate_below_delay_estimate_is_loss_limited() {
        // The same loss controller against a 20 Mbps delay-based estimate limits the estimate.
        let delay = Bitrate::mbps(20);
        let bwe = loss_controller_out_of_delay_based(delay);
        let loss = bwe.loss_controller.loss_based_result();
        println!("loss-based {loss:?}");
        let loss_estimate = loss.bandwidth_estimate.unwrap();
        assert!(loss_estimate < delay);
        assert_eq!(bwe.last_estimate(), Some(loss_estimate));
        assert!(bwe.is_loss_limited());
        assert!(matches!(
            bwe.bandwidth_limited_cause(),
            BandwidthLimitedCause::LossLimitedBwe | BandwidthLimitedCause::LossLimitedBweIncreasing
        ));
    }

    #[test]
    fn bitrate_targets_survive_estimator_reset() {
        let mut bwe = Bwe::new(Bitrate::kbps(300));
        bwe.set_bitrate(Bitrate::kbps(800), Bitrate::mbps(3));
        bwe.reset(Bitrate::kbps(200));
        assert_eq!(bwe.minimum_bitrate(), Bitrate::kbps(800));
        assert_eq!(bwe.desired_bitrate, Bitrate::mbps(3));
        assert_eq!(bwe.last_estimate(), Some(Bitrate::kbps(200)));
    }

    #[test]
    fn minimum_is_clamped_to_desired_and_can_be_cleared() {
        let mut bwe = Bwe::new(Bitrate::kbps(300));
        bwe.set_bitrate(Bitrate::mbps(2), Bitrate::mbps(1));
        assert_eq!(bwe.minimum_bitrate(), Bitrate::mbps(1));
        assert_eq!(bwe.desired_bitrate, Bitrate::mbps(1));
        bwe.set_bitrate(Bitrate::ZERO, Bitrate::mbps(1));
        assert_eq!(bwe.minimum_bitrate(), Bitrate::ZERO);
        assert_eq!(bwe.desired_bitrate, Bitrate::mbps(1));
    }
}
