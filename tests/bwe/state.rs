//! The estimator state snapshot of `Bwe::state`.

use std::time::{Duration, Instant};

use netem::{DataSize, NetemConfig};
use str0m::bwe::Bitrate;
use str0m::{Rtc, RtcError};

use crate::common::{BweTestContext, connect_with_bwe_smoothing, init_crypto_default, init_log};

#[test]
fn state_follows_the_estimator() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    let mut rtc = Rtc::builder().build(Instant::now());
    assert_eq!(rtc.bwe().state(), None, "no state without BWE");

    let (mut l, mut r) = connect_with_bwe_smoothing(Bitrate::mbps(1), Bitrate::mbps(10), false);
    let mut ctx = BweTestContext::new(&mut l, &mut r);
    let link = NetemConfig::new()
        .latency(Duration::from_millis(20))
        .link(Bitrate::mbps(5), DataSize::kbytes(100))
        .seed(42);
    l.set_netem(link);
    r.set_netem(link);
    ctx.set_media_send_rate(Bitrate::mbps(1));
    let event = ctx
        .run_for_duration(&mut l, &mut r, Duration::from_secs(5))?
        .expect("an estimate");

    let state = l.bwe().state().expect("BWE is enabled");
    println!("{state:?}");
    // With smoothing off the event carries the unsmoothed estimate.
    assert_eq!(state.estimate, Some(event));
    let acked = state.acked.expect("an acked rate");
    assert!(
        acked > Bitrate::kbps(800) && acked < Bitrate::kbps(1_300),
        "acked {acked} for 1 Mbps of media"
    );
    // 1 Mbps is below 65 % of the estimate.
    assert!(state.application_limited);
    assert!(!state.loss_limited && !state.increase_using_padding);
    assert!(state.delay_estimate.is_some());
    Ok(())
}
