//! Tests extracted from `ball.rs`.
use bvh::ball::Ball;
use bvh_testutil::TPoint3;

#[test]
fn ball_contains() {
    let ball = Ball::new(TPoint3::new(3.0, 4.0, 5.0), 1.5);

    // Ball should contain its own center.
    assert!(ball.contains(&ball.center));

    // Test some manually-selected points.
    let just_inside = TPoint3::new(3.04605, 3.23758, 3.81607);
    let just_outside = TPoint3::new(3.06066, 3.15813, 3.70917);
    assert!(ball.contains(&just_inside));
    assert!(!ball.contains(&just_outside));
}
