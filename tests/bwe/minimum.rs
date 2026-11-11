//! Minimum transmit rate with fixed media layers and unchanged desired bandwidth.

use std::time::Duration;

use netem::{DataSize, NetemConfig};
use str0m::RtcError;
use str0m::bwe::Bitrate;

use crate::common::{BweTestContext, connect_with_bwe_smoothing, init_crypto_default, init_log};

/// Time to show 2 Mbps again after `good` of a 5 Mbps link and 2 s of a 1 Mbps link with a
/// `buffer`, for a sender that drops to a fixed 500 kbps layer with `minimum`.
fn recovery_time(minimum: Bitrate, buffer: DataSize, good: Duration) -> Result<Duration, RtcError> {
    let desired = Bitrate::mbps(5);
    // The unsmoothed estimate: the 3 s mean averages the distinct estimates of the window
    // regardless of how long each held, so when it crosses 2 Mbps depends on how many small
    // steps the estimate took, not only on how fast it recovered.
    let (mut l, mut r) = connect_with_bwe_smoothing(Bitrate::mbps(2), desired, false);
    let mut ctx = BweTestContext::new(&mut l, &mut r);

    let good_link = NetemConfig::new()
        .latency(Duration::from_millis(20))
        .link(Bitrate::mbps(5), DataSize::kbytes(100))
        .seed(42);
    l.set_netem(good_link);
    r.set_netem(good_link);
    ctx.run_for_duration(&mut l, &mut r, good)?;

    r.set_netem(
        NetemConfig::new()
            .latency(Duration::from_millis(40))
            .link(Bitrate::mbps(1), buffer)
            .seed(42),
    );
    ctx.run_for_duration(&mut l, &mut r, Duration::from_secs(2))?;

    // The allocator drops to a fixed layer; desired bandwidth remains unchanged.
    ctx.set_media_send_rate(Bitrate::kbps(500));
    l.bwe().set_bitrate(minimum, desired);
    r.set_netem(good_link);
    // Measure time to regain a 2 Mbps layer, rather than comparing eventual
    // estimates after both senders have had time to run periodic probes.
    for half_seconds in 1..=20 {
        let estimate = ctx
            .run_for_duration(&mut l, &mut r, Duration::from_millis(500))?
            .expect("a bandwidth estimate after recovery");
        if estimate >= Bitrate::mbps(2) {
            return Ok(Duration::from_millis(half_seconds * 500));
        }
    }
    panic!("did not recover to 2 Mbps within 10 seconds with minimum {minimum}");
}

/// Recovery times without and with a 2 Mbps minimum, with the link recovering at 10 points
/// spread over one 5 s periodic ALR probe interval (the length of the 5 Mbps phase).
fn recovery_times(buffer: DataSize) -> Vec<(Duration, Duration, Duration)> {
    let phases = (0..10).map(|step| Duration::from_millis(5000 + 500 * step));
    std::thread::scope(|scope| {
        let runs: Vec<_> = phases
            .map(|good| {
                scope.spawn(move || {
                    let automatic = recovery_time(Bitrate::ZERO, buffer, good)?;
                    let sustained = recovery_time(Bitrate::mbps(2), buffer, good)?;
                    Ok::<_, RtcError>((good, automatic, sustained))
                })
            })
            .collect();
        runs.into_iter()
            .map(|run| run.join().expect("recovery run").expect("recovery run"))
            .collect()
    })
}

#[test]
fn minimum_improves_recovery_with_a_fixed_layer() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();
    // A 200 kB buffer: the 1 Mbps phase loses fewer packets, the loss-based estimate does not
    // limit, and the estimate is delay based when the link recovers. A fixed layer below
    // 1 / 1.5 of the estimate does not let the delay-based estimate increase; the minimum's
    // padding (not media, so the sender stays in ALR) raises the acked rate to the estimate,
    // so it can. Without the minimum the estimate waits for a probe, or in some phases for
    // the loss controller's recovery padding, and when the probe comes depends on the
    // history of the run: if it comes early, both senders recover with it. Hence the sweep
    // over one periodic probe interval: the minimum never slows the recovery, and it
    // shortens it where the automatic sender has to wait.
    let times = recovery_times(DataSize::kbytes(200));
    for &(good, automatic, sustained) in &times {
        println!(
            "delay limited, {good:?} at 5 Mbps: automatic {automatic:?}, with the minimum {sustained:?}"
        );
        assert!(
            sustained < Duration::from_secs(8),
            "recovery: {sustained:?}"
        );
        assert!(
            sustained <= automatic,
            "{good:?} at 5 Mbps: sustained traffic took {sustained:?}, automatic padding took {automatic:?}"
        );
    }
    assert!(
        times
            .iter()
            .any(|(_, automatic, sustained)| sustained < automatic),
        "the minimum shortened no recovery"
    );

    // A 50 kB buffer: the 1 Mbps phase loses packets, so the estimate is loss limited when
    // the link recovers, and the loss controller pads up to its recovering estimate on its
    // own. The minimum then adds no traffic the automatic recovery lacks; it must not slow
    // the recovery.
    let loss_limited = DataSize::kbytes(50);
    let good = Duration::from_secs(5);
    let automatic = recovery_time(Bitrate::ZERO, loss_limited, good)?;
    let sustained = recovery_time(Bitrate::mbps(2), loss_limited, good)?;
    println!("loss limited: automatic {automatic:?}, with the minimum {sustained:?}");
    assert!(
        sustained <= automatic,
        "sustained traffic took {sustained:?}, automatic padding took {automatic:?}"
    );
    Ok(())
}

#[test]
fn minimum_fills_media_gaps_and_can_be_lowered_without_probing() -> Result<(), RtcError> {
    use std::time::Instant;

    use str0m::{Event, Rtc};

    use crate::common::{TestRtc, connect_l_r_with_rtc};

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

    init_log();
    init_crypto_default();
    let start = Instant::now();
    let sender = Rtc::builder()
        .set_rtp_mode(true)
        .enable_bwe(Some(Bitrate::mbps(1)))
        .set_stats_interval(Some(Duration::from_millis(100)))
        .build(start);
    let receiver = Rtc::builder().set_rtp_mode(true).build(start);
    let (mut l, mut r) = connect_l_r_with_rtc(sender, receiver);
    l.set_forced_time_advance(Duration::from_micros(100));
    r.set_forced_time_advance(Duration::from_micros(100));
    let mut ctx = BweTestContext::new(&mut l, &mut r);
    ctx.set_media_send_rate(Bitrate::kbps(150));
    let desired = Bitrate::mbps(1);
    l.bwe().set_bitrate(Bitrate::ZERO, desired);
    ctx.run_for_duration(&mut l, &mut r, Duration::from_secs(2))?;

    let offset = l.events.len();
    l.bwe().set_bitrate(Bitrate::kbps(500), desired);
    ctx.run_for_duration(&mut l, &mut r, Duration::from_millis(800))?;
    let padded = wire_rate(&l, offset);
    assert!(padded > Bitrate::kbps(400), "transmit floor: {padded}");
    assert!(padded < Bitrate::kbps(650), "total traffic: {padded}");

    // Media replaces padding rather than adding to it.
    let media_offset = l.events.len();
    ctx.set_media_send_rate(Bitrate::kbps(600));
    ctx.run_for_duration(&mut l, &mut r, Duration::from_millis(800))?;
    let media = wire_rate(&l, media_offset);
    assert!(media > Bitrate::kbps(550), "media rate: {media}");
    assert!(
        media < Bitrate::kbps(800),
        "padding must not add 500 kbps: {media}"
    );

    let lower_offset = l.events.len();
    ctx.set_media_send_rate(Bitrate::kbps(150));
    l.bwe().set_bitrate(Bitrate::ZERO, desired);
    ctx.run_for_duration(&mut l, &mut r, Duration::from_millis(800))?;
    let lowered = wire_rate(&l, lower_offset);
    assert!(lowered < Bitrate::kbps(250), "lowered traffic: {lowered}");
    assert!(
        l.events[offset..]
            .iter()
            .all(|(_, e)| !matches!(e, Event::Probe(_))),
        "changing the minimum must not force a probe"
    );
    // An explicit minimum cannot sustain traffic after the sending media is removed.
    l.bwe().set_bitrate(Bitrate::kbps(500), desired);
    l.direct_api().remove_media("vid".into());
    ctx.set_media_send_rate(Bitrate::ZERO);
    let removed_offset = l.events.len();
    ctx.run_for_duration(&mut l, &mut r, Duration::from_millis(800))?;
    let removed = wire_rate(&l, removed_offset);
    assert!(
        removed < Bitrate::kbps(50),
        "traffic without sending media: {removed}"
    );
    Ok(())
}
