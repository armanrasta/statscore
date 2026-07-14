//! Quick wall-clock microbench (no Criterion). Run with:
//! `cargo run -p statscore-distributions --release --example microbench`

use std::hint::black_box;
use std::time::Instant;

use rand::rng;
use statscore_common::{ContinuousDistribution, DiscreteDistribution};
use statscore_distributions::{
    Beta, Binomial, ChiSquared, Exponential, Gamma, Normal, Poisson, StudentT,
};

fn time_ns(iters: u32, mut f: impl FnMut()) -> f64 {
    // warmup
    for _ in 0..iters / 10 + 1 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e9 / f64::from(iters)
}

fn main() {
    let n = Normal::standard();
    let g = Gamma::new(2.5, 1.5).unwrap();
    let b = Beta::new(2.0, 5.0).unwrap();
    let t = StudentT::new(10.0).unwrap();
    let c = ChiSquared::new(5.0).unwrap();
    let e = Exponential::new(1.5).unwrap();
    let bn = Binomial::new(20, 0.3).unwrap();
    let p = Poisson::new(4.0).unwrap();

    println!("statscore-distributions microbench (release, ns/op median-ish)");
    println!("{:<28} {:>12}", "op", "ns/op");
    println!("{}", "-".repeat(42));

    let rows: &[(&str, u32, Box<dyn FnMut()>)] = &[];
    let _ = rows; // placate; we call time_ns directly below

    macro_rules! row {
        ($name:expr, $iters:expr, $body:expr) => {{
            let ns = time_ns($iters, $body);
            println!("{:<28} {:>10.1}", $name, ns);
        }};
    }

    row!("Normal.pdf", 5_000_000, || {
        black_box(n.pdf(black_box(0.5)));
    });
    row!("Normal.cdf", 5_000_000, || {
        black_box(n.cdf(black_box(1.96)));
    });
    row!("Normal.ppf", 2_000_000, || {
        black_box(n.ppf(black_box(0.975)).unwrap());
    });
    row!("Gamma.cdf", 500_000, || {
        black_box(g.cdf(black_box(3.0)));
    });
    row!("Gamma.ppf", 50_000, || {
        black_box(g.ppf(black_box(0.5)).unwrap());
    });
    row!("Beta.cdf", 500_000, || {
        black_box(b.cdf(black_box(0.3)));
    });
    row!("Beta.ppf", 50_000, || {
        black_box(b.ppf(black_box(0.5)).unwrap());
    });
    row!("StudentT.cdf", 200_000, || {
        black_box(t.cdf(black_box(1.5)));
    });
    row!("StudentT.ppf", 50_000, || {
        black_box(t.ppf(black_box(0.95)).unwrap());
    });
    row!("ChiSquared.ppf", 50_000, || {
        black_box(c.ppf(black_box(0.95)).unwrap());
    });
    row!("Exponential.cdf", 5_000_000, || {
        black_box(e.cdf(black_box(1.0)));
    });
    row!("Binomial.pmf", 1_000_000, || {
        black_box(bn.pmf(black_box(6)));
    });
    row!("Binomial.cdf", 200_000, || {
        black_box(bn.cdf(black_box(6)));
    });
    row!("Poisson.pmf", 1_000_000, || {
        black_box(p.pmf(black_box(4)));
    });
    row!("Poisson.cdf", 200_000, || {
        black_box(p.cdf(black_box(4)));
    });

    // Batch 10k in Rust
    let xs: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.001 - 5.0).collect();
    let ns = time_ns(2_000, || {
        let _: Vec<_> = xs.iter().map(|&x| n.pdf(black_box(x))).collect();
    });
    println!("{:<28} {:>10.1}  ({:.1} ns/elem)", "Normal.pdf×10k", ns, ns / 10_000.0);

    let ns = time_ns(2_000, || {
        let _: Vec<_> = xs.iter().map(|&x| n.cdf(black_box(x))).collect();
    });
    println!("{:<28} {:>10.1}  ({:.1} ns/elem)", "Normal.cdf×10k", ns, ns / 10_000.0);

    let ns = time_ns(200, || {
        black_box(n.sample(&mut rng(), black_box(100_000)));
    });
    println!(
        "{:<28} {:>10.1}  ({:.1} ns/sample)",
        "Normal.rvs(100k)",
        ns,
        ns / 100_000.0
    );
}
