//! Basic tour of `statscore-fuzzy`: sets, logic, and statistics.

use statscore_common::Result;
use statscore_fuzzy::logic::FuzzyLogic;
use statscore_fuzzy::sets::{TrapezoidalFuzzyNumber, TriangularFuzzyNumber};
use statscore_fuzzy::statistics::{fuzzy_correlation, fuzzy_mean, fuzzy_variance};
use statscore_fuzzy::traits::{FuzzyNumber, FuzzySet};

fn main() -> Result<()> {
    println!("== Fuzzy sets ==");
    let warm = TriangularFuzzyNumber::new(18.0, 22.0, 26.0)?;
    println!("warm ≈ 22°C  (triangular {warm:?})");
    for x in [18.0, 20.0, 22.0, 24.0, 30.0] {
        println!("  μ_warm({x:>4.1}) = {:.2}", warm.membership(x));
    }
    println!("  core={:?} support={:?}", warm.core(), warm.support());
    println!("  0.5-cut = {:?}", warm.alpha_cut(0.5));
    println!(
        "  defuzzify: COG={:.3} MOM={:.3}",
        warm.defuzzify_cog(),
        warm.defuzzify_mom()
    );

    let comfortable = TrapezoidalFuzzyNumber::new(19.0, 21.0, 24.0, 26.0)?;
    println!("\ncomfortable (trapezoid {comfortable:?})");
    println!("  μ(22.5) = {:.2}", comfortable.membership(22.5));
    println!("  COG     = {:.3}", comfortable.defuzzify_cog());

    println!("\n== Fuzzy logic ==");
    let (a, b) = (0.7, 0.4);
    println!("  AND(min)        = {:.2}", FuzzyLogic::fuzzy_and_min(a, b));
    println!("  AND(product)    = {:.2}", FuzzyLogic::fuzzy_and_product(a, b));
    println!("  OR(max)         = {:.2}", FuzzyLogic::fuzzy_or_max(a, b));
    println!("  OR(sum)         = {:.2}", FuzzyLogic::fuzzy_or_sum(a, b));
    println!("  NOT(0.7)        = {:.2}", FuzzyLogic::fuzzy_not(a));

    println!("\n== Fuzzy statistics ==");
    let measurements = [
        TriangularFuzzyNumber::new(4.5, 5.0, 5.5)?,
        TriangularFuzzyNumber::new(4.8, 5.1, 5.4)?,
        TriangularFuzzyNumber::new(4.9, 5.0, 5.1)?,
    ];
    let mean = fuzzy_mean(&measurements)?;
    println!(
        "  fuzzy mean = [{:.3}, {:.3}, {:.3}]  (COG={:.3})",
        mean.a,
        mean.m,
        mean.b,
        mean.defuzzify_cog()
    );
    println!("  fuzzy variance = {:.5}", fuzzy_variance(&measurements)?);

    let x = [
        TriangularFuzzyNumber::new(1.0, 2.0, 3.0)?,
        TriangularFuzzyNumber::new(3.0, 4.0, 5.0)?,
        TriangularFuzzyNumber::new(5.0, 6.0, 7.0)?,
    ];
    let y = [
        TriangularFuzzyNumber::new(2.0, 3.0, 4.0)?,
        TriangularFuzzyNumber::new(4.0, 5.0, 6.0)?,
        TriangularFuzzyNumber::new(6.0, 7.0, 8.0)?,
    ];
    println!("  fuzzy correlation = {:.3}", fuzzy_correlation(&x, &y)?);

    Ok(())
}
