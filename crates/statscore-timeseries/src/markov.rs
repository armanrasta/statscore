//! Discrete-time Markov chains for categorical state sequences.

use rand::{Rng, RngExt};
use statscore_common::{Result, StatsError};
use statscore_linalg::matrix::{square_from_row_slice, vector_from_slice};
use statscore_linalg::solve::solve_linear_system;

/// Discrete-time Markov chain estimated from a state sequence.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    /// Number of states `0..n_states`.
    pub n_states: usize,
    /// Row-stochastic transition matrix `P[i][j] = P(X_{t+1}=j | X_t=i)`.
    transition: Vec<Vec<f64>>,
}

impl MarkovChain {
    /// Estimate transition counts → row-normalized `P`.
    ///
    /// States must lie in `0..n_states`.
    ///
    /// # Errors
    /// Returns an error if the sequence is too short or contains an out-of-range state.
    ///
    /// # Example
    /// ```
    /// use statscore_timeseries::markov::MarkovChain;
    /// let states = vec![0, 1, 0, 1, 1, 0];
    /// let mc = MarkovChain::fit(&states, 2).unwrap();
    /// assert_eq!(mc.n_states, 2);
    /// let p = mc.predict_proba(0).unwrap();
    /// assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    /// ```
    pub fn fit(states: &[usize], n_states: usize) -> Result<Self> {
        if n_states == 0 {
            return Err(StatsError::domain("n_states must be >= 1"));
        }
        if states.len() < 2 {
            return Err(StatsError::insufficient_data(2, states.len()));
        }
        let mut counts = vec![vec![0.0; n_states]; n_states];
        for w in states.windows(2) {
            let (i, j) = (w[0], w[1]);
            if i >= n_states || j >= n_states {
                return Err(StatsError::domain(format!(
                    "markov state out of range: got {i},{j}, n_states={n_states}"
                )));
            }
            counts[i][j] += 1.0;
        }
        let mut transition = vec![vec![0.0; n_states]; n_states];
        for i in 0..n_states {
            let row_sum: f64 = counts[i].iter().sum();
            if row_sum == 0.0 {
                // absorbing self-loop if never observed
                transition[i][i] = 1.0;
            } else {
                for j in 0..n_states {
                    transition[i][j] = counts[i][j] / row_sum;
                }
            }
        }
        Ok(Self {
            n_states,
            transition,
        })
    }

    /// Borrow the transition matrix as nested rows.
    #[must_use]
    pub fn transition(&self) -> &[Vec<f64>] {
        &self.transition
    }

    /// Flat row-major transition matrix (length `n²`).
    #[must_use]
    pub fn transition_flat(&self) -> Vec<f64> {
        self.transition.iter().flatten().copied().collect()
    }

    /// Stationary distribution π (solve `(Pᵀ − I)π = 0` with Σπ=1).
    ///
    /// # Errors
    /// Returns an error if the linear system is singular.
    pub fn stationary(&self) -> Result<Vec<f64>> {
        let n = self.n_states;
        // Build A π = b with A = Pᵀ - I, replace last row with ones
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                // (Pᵀ)_{ij} = P_{ji}
                a[i * n + j] = self.transition[j][i];
            }
            a[i * n + i] -= 1.0;
        }
        for j in 0..n {
            a[(n - 1) * n + j] = 1.0;
        }
        let mut b = vec![0.0; n];
        b[n - 1] = 1.0;
        let mat = square_from_row_slice(n, &a)?;
        let rhs = vector_from_slice(&b);
        let pi = solve_linear_system(&mat, &rhs)?;
        Ok((0..n).map(|i| pi.get(i).max(0.0)).collect())
    }

    /// One-step predictive distribution from `current` state.
    ///
    /// # Errors
    /// Returns an error if `current` is out of range.
    pub fn predict_proba(&self, current: usize) -> Result<Vec<f64>> {
        if current >= self.n_states {
            return Err(StatsError::domain(format!(
                "current state {current} >= n_states {}",
                self.n_states
            )));
        }
        Ok(self.transition[current].clone())
    }

    /// Simulate a path of length `n` starting from `start`.
    pub fn sample_path<R: Rng + ?Sized>(
        &self,
        start: usize,
        n: usize,
        rng: &mut R,
    ) -> Result<Vec<usize>> {
        if start >= self.n_states {
            return Err(StatsError::domain("start state out of range"));
        }
        if n == 0 {
            return Ok(vec![]);
        }
        let mut out = Vec::with_capacity(n);
        let mut s = start;
        out.push(s);
        for _ in 1..n {
            let u: f64 = rng.random();
            let mut cum = 0.0;
            let mut next = self.n_states - 1;
            for (j, &p) in self.transition[s].iter().enumerate() {
                cum += p;
                if u <= cum {
                    next = j;
                    break;
                }
            }
            s = next;
            out.push(s);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::{RngExt, rng};

    #[test]
    fn markov_two_state() {
        // Generate from known P
        let mut states = vec![0usize];
        let p01 = 0.3;
        let p10 = 0.4;
        let mut r = rng();
        for _ in 0..5000 {
            let s = *states.last().unwrap();
            let u: f64 = RngExt::random(&mut r);
            let next = if s == 0 {
                if u < p01 { 1 } else { 0 }
            } else if u < p10 {
                0
            } else {
                1
            };
            states.push(next);
        }
        let mc = MarkovChain::fit(&states, 2).unwrap();
        assert_relative_eq!(mc.transition()[0][1], p01, epsilon = 0.05);
        assert_relative_eq!(mc.transition()[1][0], p10, epsilon = 0.05);
        let pi = mc.stationary().unwrap();
        assert_relative_eq!(pi.iter().sum::<f64>(), 1.0, epsilon = 1e-8);
    }
}
