//! Criterion benches for core distribution operations.
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::rng;
use statscore_common::{ContinuousDistribution, DiscreteDistribution};
use statscore_distributions::{
    Beta, Binomial, ChiSquared, Exponential, Gamma, Normal, Poisson, StudentT,
};

fn bench_normal(c: &mut Criterion) {
    let n = Normal::standard();
    let mut group = c.benchmark_group("normal");
    group.throughput(Throughput::Elements(1));

    group.bench_function("pdf", |b| b.iter(|| n.pdf(black_box(0.5))));
    group.bench_function("cdf", |b| b.iter(|| n.cdf(black_box(1.96))));
    group.bench_function("ppf", |b| b.iter(|| n.ppf(black_box(0.975)).unwrap()));
    group.bench_function("log_pdf", |b| b.iter(|| n.log_pdf(black_box(-0.3))));

    for &size in &[1_000usize, 100_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("rvs", size), &size, |b, &size| {
            b.iter(|| n.sample(&mut rng(), black_box(size)));
        });
    }
    group.finish();
}

fn bench_batch_eval(c: &mut Criterion) {
    let n = Normal::standard();
    let xs: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.001 - 5.0).collect();
    let mut group = c.benchmark_group("normal_batch");
    group.throughput(Throughput::Elements(xs.len() as u64));

    group.bench_function("pdf_10k", |b| {
        b.iter(|| {
            xs.iter()
                .map(|&x| n.pdf(black_box(x)))
                .collect::<Vec<_>>()
        })
    });
    group.bench_function("cdf_10k", |b| {
        b.iter(|| {
            xs.iter()
                .map(|&x| n.cdf(black_box(x)))
                .collect::<Vec<_>>()
        })
    });
    group.finish();
}

fn bench_others(c: &mut Criterion) {
    let gamma = Gamma::new(2.5, 1.5).unwrap();
    let beta = Beta::new(2.0, 5.0).unwrap();
    let chi2 = ChiSquared::new(5.0).unwrap();
    let t = StudentT::new(10.0).unwrap();
    let exp = Exponential::new(1.5).unwrap();
    let binom = Binomial::new(20, 0.3).unwrap();
    let pois = Poisson::new(4.0).unwrap();

    let mut group = c.benchmark_group("distributions");
    group.throughput(Throughput::Elements(1));

    group.bench_function("gamma/cdf", |b| b.iter(|| gamma.cdf(black_box(3.0))));
    group.bench_function("gamma/ppf", |b| {
        b.iter(|| gamma.ppf(black_box(0.5)).unwrap())
    });
    group.bench_function("beta/cdf", |b| b.iter(|| beta.cdf(black_box(0.3))));
    group.bench_function("beta/ppf", |b| b.iter(|| beta.ppf(black_box(0.5)).unwrap()));
    group.bench_function("chi2/cdf", |b| b.iter(|| chi2.cdf(black_box(5.0))));
    group.bench_function("chi2/ppf", |b| b.iter(|| chi2.ppf(black_box(0.95)).unwrap()));
    group.bench_function("student_t/cdf", |b| b.iter(|| t.cdf(black_box(1.5))));
    group.bench_function("student_t/ppf", |b| {
        b.iter(|| t.ppf(black_box(0.95)).unwrap())
    });
    group.bench_function("exponential/cdf", |b| b.iter(|| exp.cdf(black_box(1.0))));
    group.bench_function("binomial/pmf", |b| b.iter(|| binom.pmf(black_box(6))));
    group.bench_function("binomial/cdf", |b| b.iter(|| binom.cdf(black_box(6))));
    group.bench_function("poisson/pmf", |b| b.iter(|| pois.pmf(black_box(4))));
    group.bench_function("poisson/cdf", |b| b.iter(|| pois.cdf(black_box(4))));
    group.finish();
}

criterion_group!(benches, bench_normal, bench_batch_eval, bench_others);
criterion_main!(benches);
