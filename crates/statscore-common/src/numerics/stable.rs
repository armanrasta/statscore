/// Numerically stable log(sum(exp(x)))
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Numerically stable softmax
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let lse = log_sum_exp(values);
    values.iter().map(|v| (v - lse).exp()).collect()
}

/// log(1 + exp(x)) without overflow
pub fn log1pexp(x: f64) -> f64 {
    if x <= -37.0 {
        x.exp()
    } else if x <= 18.0 {
        (1.0 + x.exp()).ln()
    } else if x <= 33.3 {
        x + (-x).exp()
    } else {
        x
    }
}

/// log(exp(a) + exp(b)) without overflow
pub fn log_add(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }
    if a > b {
        a + log1pexp(b - a)
    } else {
        b + log1pexp(a - b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_log_sum_exp_basic() {
        let vals = vec![1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()];
        let result = log_sum_exp(&vals);
        assert_relative_eq!(result, 6.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sm = softmax(&vals);
        let sum: f64 = sm.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_log1pexp_small() {
        assert_relative_eq!(log1pexp(-100.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_log1pexp_large() {
        assert_relative_eq!(log1pexp(100.0), 100.0, epsilon = 1e-12);
    }
}