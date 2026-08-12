//! Demo: baselines, ETS, ARIMA, Prophet-style, Markov, stationarity, decompose.

use rand::rng;
use statscore_timeseries::arima::ArimaModel;
use statscore_timeseries::baselines::{drift, naive};
use statscore_timeseries::decomposition::{DecomposeModel, classical_decompose};
use statscore_timeseries::ets::{EtsFit, EtsModel};
use statscore_timeseries::forecast::Forecaster;
use statscore_timeseries::markov::MarkovChain;
use statscore_timeseries::prophet::{ProphetStyleModel, ProphetStyleSpec};
use statscore_timeseries::stationarity::{KpssKind, adf_test, kpss_test};

fn main() {
    let x: Vec<f64> = (0..48)
        .map(|i| {
            let t = i as f64;
            10.0 + 0.2 * t + 2.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin()
        })
        .collect();

    println!("naive  h=3  {:?}", naive(&x, 3).unwrap().point);
    println!("drift  h=3  {:?}", drift(&x, 3).unwrap().point);

    let ets = EtsFit::fit(&x, EtsModel::Aaa, Some(12)).unwrap();
    println!("ETS AAA h=3 {:?}", ets.forecast(3).unwrap().point);

    let arima = ArimaModel::fit(&x, 1, 1, 0).unwrap();
    println!(
        "ARIMA(1,1,0) h=3 {:?}  AIC={:.3}",
        arima.forecast(3).unwrap().point,
        arima.aic()
    );

    let t: Vec<f64> = (0..x.len()).map(|i| i as f64).collect();
    let spec = ProphetStyleSpec {
        period: 12.0,
        n_changepoints: 3,
        fourier_order: 2,
    };
    let prophet = ProphetStyleModel::fit(&t, &x, &spec).unwrap();
    let future = vec![48.0, 49.0, 50.0];
    println!(
        "Prophet-style {:?}",
        prophet.predict(&future).unwrap().point
    );

    let adf = adf_test(&x, Some(2)).unwrap();
    let kpss = kpss_test(&x, KpssKind::Level).unwrap();
    println!(
        "ADF stat={:.3} (n={}, lags={})  KPSS={:.4}",
        adf.statistic, adf.nobs, adf.lags, kpss.statistic
    );

    let d = classical_decompose(&x, 12, DecomposeModel::Additive).unwrap();
    println!("decompose: seasonal[0..3]={:?}", &d.seasonal[..3]);

    let states = vec![0, 1, 0, 1, 1, 0, 0, 1];
    let mc = MarkovChain::fit(&states, 2).unwrap();
    let path = mc.sample_path(0, 8, &mut rng()).unwrap();
    println!(
        "Markov P[0]={:?}  sample={path:?}",
        mc.predict_proba(0).unwrap()
    );
}
