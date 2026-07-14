//! Continuous distributions.

mod beta;
mod chi_squared;
mod exponential;
mod f_distribution;
mod gamma;
mod normal;
mod student_t;
mod uniform;

pub use beta::Beta;
pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use f_distribution::FDistribution;
pub use gamma::Gamma;
pub use normal::Normal;
pub use student_t::StudentT;
pub use uniform::Uniform;
