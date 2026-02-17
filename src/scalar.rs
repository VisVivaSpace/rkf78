//! Scalar type traits for generic numeric computation.
//!
//! Provides `Float` (real numbers) and `Scalar` (real or complex) traits
//! that allow the solver to work with f32, f64, and (future) complex types.

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// A real floating-point type (f32 or f64).
///
/// Used for time, step sizes, tolerances, error estimates — quantities
/// that are always real even when the state is complex.
pub trait Float:
    Copy
    + Clone
    + Debug
    + Display
    + PartialOrd
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
    + Scalar<Real = Self>
{
    /// The multiplicative identity.
    const ONE: Self;
    /// Two.
    const TWO: Self;
    /// One half.
    const HALF: Self;
    /// Positive infinity.
    const INFINITY: Self;

    /// Convert from f64 (used for Butcher tableau coefficients).
    fn from_f64(v: f64) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Sign: -1, 0, or 1.
    fn signum(self) -> Self;

    /// Clamp to [min, max].
    fn clamp(self, min: Self, max: Self) -> Self;

    /// Maximum of two values.
    fn max(self, other: Self) -> Self;

    /// Raise to a real power.
    fn powf(self, exp: Self) -> Self;

    /// Returns true if finite (not NaN or infinity).
    fn is_finite(self) -> bool;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Sine.
    fn sin(self) -> Self;

    /// Cosine.
    fn cos(self) -> Self;
}

/// A scalar type that can be used as a state component.
///
/// For real types (f32, f64), this is identical to `Float`.
/// For complex types, `Real` is the underlying float and `norm` gives the modulus.
pub trait Scalar:
    Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + MulAssign
{
    /// The underlying real type.
    type Real: Float;

    /// The additive identity.
    const ZERO: Self;

    /// Create from a real value.
    fn from_real(r: Self::Real) -> Self;

    /// The norm (absolute value for reals, modulus for complex).
    fn norm(self) -> Self::Real;

    /// Multiply by a real scalar (avoids full complex multiply).
    fn mul_real(self, r: Self::Real) -> Self;
}

macro_rules! impl_float_scalar {
    ($ty:ty) => {
        impl Float for $ty {
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const INFINITY: Self = <$ty>::INFINITY;

            #[inline]
            fn from_f64(v: f64) -> Self {
                v as Self
            }

            #[inline]
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }

            #[inline]
            fn signum(self) -> Self {
                <$ty>::signum(self)
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                <$ty>::clamp(self, min, max)
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                <$ty>::max(self, other)
            }

            #[inline]
            fn powf(self, exp: Self) -> Self {
                <$ty>::powf(self, exp)
            }

            #[inline]
            fn is_finite(self) -> bool {
                <$ty>::is_finite(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }

            #[inline]
            fn sin(self) -> Self {
                <$ty>::sin(self)
            }

            #[inline]
            fn cos(self) -> Self {
                <$ty>::cos(self)
            }
        }

        impl Scalar for $ty {
            type Real = Self;
            const ZERO: Self = 0.0;

            #[inline]
            fn from_real(r: Self) -> Self {
                r
            }

            #[inline]
            fn norm(self) -> Self {
                <$ty>::abs(self)
            }

            #[inline]
            fn mul_real(self, r: Self) -> Self {
                self * r
            }
        }
    };
}

impl_float_scalar!(f32);
impl_float_scalar!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_float_basics() {
        assert_eq!(<f64 as Scalar>::ZERO, 0.0);
        assert_eq!(<f64 as Float>::ONE, 1.0);
        assert_eq!(<f64 as Float>::from_f64(3.14), 3.14);
        assert_eq!(Float::abs(-2.5_f64), 2.5);
        assert_eq!(Float::signum(-2.5_f64), -1.0);
        assert!(!<f64 as Float>::INFINITY.is_finite());
        assert!(1.0_f64.is_finite());
    }

    #[test]
    fn test_f32_float_basics() {
        assert_eq!(<f32 as Scalar>::ZERO, 0.0_f32);
        assert_eq!(<f32 as Float>::ONE, 1.0_f32);
        let pi_f32 = <f32 as Float>::from_f64(std::f64::consts::PI);
        assert!((pi_f32 - std::f32::consts::PI).abs() < 1e-7);
        assert!(!<f32 as Float>::INFINITY.is_finite());
    }

    #[test]
    fn test_scalar_f64() {
        let x: f64 = 3.0;
        assert_eq!(x.norm(), 3.0);
        assert_eq!((-x).norm(), 3.0);
        assert_eq!(x.mul_real(2.0), 6.0);
        assert_eq!(<f64 as Scalar>::from_real(5.0), 5.0);
    }

    #[test]
    fn test_scalar_f32() {
        let x: f32 = 3.0;
        assert_eq!(x.norm(), 3.0_f32);
        assert_eq!(x.mul_real(2.0_f32), 6.0_f32);
    }

    #[test]
    fn test_from_f64_precision() {
        // f64 -> f64 should be exact
        let v: f64 = Float::from_f64(1.23456789012345e-10);
        assert_eq!(v, 1.23456789012345e-10);

        // f64 -> f32 truncates but should be close
        let v32: f32 = Float::from_f64(1.23456789012345e-10);
        assert!((v32 as f64 - 1.23456789012345e-10).abs() < 1e-17);
    }
}
