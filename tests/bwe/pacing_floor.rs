//! A pacing floor lets media that the application already limits leave at its own rate.

use std::time::{Duration, Instant};

use netem::{DataSize, NetemConfig, Probability, RandomLoss};
use str0m::bwe::Bitrate;
use str0m::{Event, Rtc, RtcError};

use crate::common::{BweTestContext, TestRtc, connect_l_r_with_rtc};
use crate::common::{init_crypto_default, init_log};

/// What L put on the wire from event `since` on, from its peer stats.
fn wire_rate(rtc: &TestRtc, since: usize) -> Bitrate {
    let mut stats = rtc.events[since..].iter().filter_map(|(_, event)| {
        if let Event::PeerStats(stats) = event {
            Some(stats)
        } else {
            None
        }
    });
    let first = stats.next().expect("first traffic sample");
    let last = stats.next_back().expect("last traffic sample");
    let bytes = DataSize::bytes((last.peer_bytes_tx - first.peer_bytes_tx) as i64);
    bytes / last.timestamp.duration_since(first.timestamp)
}

/// 1 Mbps of media against an estimate held at its 300 kbps initial value (all feedback to
/// the sender is lost), with `floor`. Returns the wire rate over 1.5 s and the estimate.
fn send_above_estimate(floor: Bitrate) -> Result<(Bitrate, Option<Bitrate>), RtcError> {
    init_log();
    init_crypto_default();
    let start = Instant::now();
    let sender = Rtc::builder()
        .set_rtp_mode(true)
        .enable_bwe(Some(Bitrate::kbps(300)))
        .set_stats_interval(Some(Duration::from_millis(100)))
        .build(start);
    let receiver = Rtc::builder().set_rtp_mode(true).build(start);
    let (mut l, mut r) = connect_l_r_with_rtc(sender, receiver);
    l.set_forced_time_advance(Duration::from_micros(100));
    r.set_forced_time_advance(Duration::from_micros(100));
    l.set_netem(NetemConfig::new().loss(RandomLoss::new(Probability::ONE)));
    let mut ctx = BweTestContext::new(&mut l, &mut r);
    l.bwe().set_bitrate(Bitrate::ZERO, Bitrate::kbps(300));
    l.bwe().set_pacing_floor(floor);
    ctx.set_media_send_rate(Bitrate::mbps(1));

    let offset = l.events.len();
    ctx.run_for_duration(&mut l, &mut r, Duration::from_millis(1_500))?;
    Ok((
        wire_rate(&l, offset),
        l.bwe().state().and_then(|s| s.estimate),
    ))
}

#[test]
fn pacing_floor_paces_media_above_the_estimate() -> Result<(), RtcError> {
    let (without, estimate) = send_above_estimate(Bitrate::ZERO)?;
    let (with, _) = send_above_estimate(Bitrate::mbps(1))?;
    println!("estimate {estimate:?}: without floor {without}, with a 1 Mbps floor {with}");
    assert_eq!(
        estimate,
        Some(Bitrate::kbps(300)),
        "no feedback, no estimate update"
    );
    // Paced at 1.1 x the estimate, the rest waits in the pacer queue.
    assert!(without < Bitrate::kbps(500), "without floor: {without}");
    // Paced at 1.1 x the floor: the media leaves as it comes.
    assert!(with > Bitrate::kbps(900), "with floor: {with}");
    Ok(())
}
