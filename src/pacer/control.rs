use crate::rtp_::Bitrate;

const PACING_FACTOR: f64 = 1.1;

/// Automatic padding target when media is active. This maintains NAT bindings, RTX state,
/// and allows ALR periodic probes to discover higher bandwidth.
const PADDING_TARGET: Bitrate = Bitrate::bps(50_000);

pub(crate) struct PacingResult {
    /// The bitrate at which the pacer may emit padding **when there is no media queued**.
    ///
    /// This value is intentionally an *absolute* target used by the pacer, not a “delta to add
    /// on top of current media”. In practice the **effective padding** over time is the
    /// difference between this target and the media actually sent, because padding is only used
    /// to fill gaps (empty queues), not to continuously top-up while media is flowing.
    pub padding_rate: Bitrate,
    pub pacing_rate: Bitrate,
}

/// Controls the pacing and padding rates.
pub(crate) struct PacerControl {}

impl PacerControl {
    pub fn new() -> Self {
        Self {}
    }

    /// `minimum` is the application's minimum transmit rate, `loss_recovery` the rate the
    /// loss controller wants padded to while it increases a loss-limited estimate (zero
    /// otherwise). As libWebRTC's `GetPacingRates`, the padding rate is the larger of the two
    /// (and the automatic target), capped by the estimate. `pacing_floor` raises the pacing
    /// rate only, as libWebRTC's `min_total_allocated_bitrate`: the pacer paces as if the
    /// estimate were at least that.
    pub fn calculate(
        &self,
        has_active_media: bool,
        estimate: Bitrate,
        is_overuse: bool,
        minimum: Bitrate,
        loss_recovery: Bitrate,
        pacing_floor: Bitrate,
    ) -> PacingResult {
        let padding_rate = if is_overuse {
            // No padding during overuse
            Bitrate::ZERO
        } else if has_active_media {
            // Sustain the requested minimum, or the loss recovery rate, without exceeding
            // estimated capacity. Keep the automatic target for NAT bindings and RTX state.
            PADDING_TARGET.max(minimum).max(loss_recovery).min(estimate)
        } else {
            // No padding when no media is being sent
            Bitrate::ZERO
        };

        // Set pacing rate to smooth out media transmission (burst avoidance).
        // Must be at least the current BWE estimate * factor, but also high enough
        // to allow the padding we want to send.
        let min_pacing_rate = estimate.max(pacing_floor) * PACING_FACTOR;
        let pacing_rate = min_pacing_rate.max(padding_rate);

        PacingResult {
            padding_rate,
            pacing_rate,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn padding_enabled_with_active_media() {
        let c = PacerControl::new();
        let estimate = Bitrate::kbps(1_000);

        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::ZERO,
            Bitrate::ZERO,
            Bitrate::ZERO,
        );

        assert_eq!(r.padding_rate, PADDING_TARGET);
    }

    #[test]
    fn no_padding_without_active_media() {
        let c = PacerControl::new();
        let estimate = Bitrate::kbps(1_000);

        let r = c.calculate(
            false,
            estimate,
            false,
            Bitrate::mbps(2),
            Bitrate::ZERO,
            Bitrate::ZERO,
        );

        assert_eq!(r.padding_rate, Bitrate::ZERO);
    }

    #[test]
    fn overuse_suppresses_padding() {
        let c = PacerControl::new();
        let estimate = Bitrate::mbps(40);

        let r = c.calculate(
            true,
            estimate,
            true,
            Bitrate::mbps(2),
            Bitrate::ZERO,
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, Bitrate::ZERO);
    }

    #[test]
    fn minimum_tracks_estimate_without_exceeding_target() {
        let c = PacerControl::new();
        let minimum = Bitrate::kbps(800);
        // Both increasing and decreasing estimates must immediately change the cap.
        for (estimate, expected) in [(450, 450), (700, 700), (1000, 800), (300, 300)] {
            let result = c.calculate(
                true,
                Bitrate::kbps(estimate),
                false,
                minimum,
                Bitrate::ZERO,
                Bitrate::ZERO,
            );
            assert_eq!(result.padding_rate, Bitrate::kbps(expected));
            assert_eq!(result.pacing_rate, Bitrate::kbps(estimate) * PACING_FACTOR);
        }
    }

    #[test]
    fn loss_recovery_raises_padding_rate() {
        let c = PacerControl::new();
        let estimate = Bitrate::kbps(1_000);

        // While the loss controller increases using padding: pad up to that rate.
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::ZERO,
            estimate,
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, estimate);
        assert!(r.pacing_rate >= r.padding_rate);

        // Never below the normal padding target, nor the minimum.
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::ZERO,
            Bitrate::kbps(20),
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, PADDING_TARGET);
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::kbps(800),
            Bitrate::kbps(500),
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, Bitrate::kbps(800));
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::kbps(500),
            Bitrate::kbps(800),
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, Bitrate::kbps(800));

        // Capped by the estimate and suppressed in overuse, as the minimum.
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::ZERO,
            Bitrate::mbps(2),
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, estimate);
        let r = c.calculate(true, estimate, true, Bitrate::ZERO, estimate, Bitrate::ZERO);
        assert_eq!(r.padding_rate, Bitrate::ZERO);

        // Back to the normal padding target after the state.
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::ZERO,
            Bitrate::ZERO,
            Bitrate::ZERO,
        );
        assert_eq!(r.padding_rate, PADDING_TARGET);
    }

    #[test]
    fn pacing_floor_raises_pacing_rate_only() {
        let c = PacerControl::new();
        let estimate = Bitrate::kbps(500);
        let floor = Bitrate::mbps(2);

        let r = c.calculate(true, estimate, false, Bitrate::mbps(1), estimate, floor);
        assert_eq!(r.pacing_rate, floor * PACING_FACTOR);
        // The padding stays capped by the estimate, not by the floor.
        assert_eq!(r.padding_rate, estimate);

        // A floor below the estimate changes nothing.
        let r = c.calculate(
            true,
            estimate,
            false,
            Bitrate::ZERO,
            Bitrate::ZERO,
            Bitrate::kbps(100),
        );
        assert_eq!(r.pacing_rate, estimate * PACING_FACTOR);
        assert_eq!(r.padding_rate, PADDING_TARGET);
    }

    #[test]
    fn automatic_padding_is_also_capped_by_estimate() {
        let c = PacerControl::new();
        for estimate in [Bitrate::ZERO, Bitrate::kbps(20)] {
            let result = c.calculate(
                true,
                estimate,
                false,
                Bitrate::ZERO,
                Bitrate::ZERO,
                Bitrate::ZERO,
            );
            assert_eq!(result.padding_rate, estimate);
        }
    }
}
