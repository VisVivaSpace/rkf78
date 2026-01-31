//! Runge-Kutta-Fehlberg 7(8) Coefficients
//!
//! Coefficients for the 13-stage embedded RK7(8) pair from:
//! Fehlberg, E. (1968). "Classical Fifth-, Sixth-, Seventh-, and 
//! Eighth-Order Runge-Kutta Formulas with Stepsize Control"
//! NASA TR R-287, Table X, pages 64-65.
//!
//! This method provides an 8th-order solution with a 7th-order
//! embedded method for error estimation and adaptive step control.

/// Number of stages in the RKF78 method
pub const STAGES: usize = 13;

/// Order of the higher-order method (used for advancing the solution)
pub const ORDER: u8 = 8;

/// Order of the embedded method (used for error estimation)
pub const EMBEDDED_ORDER: u8 = 7;

/// Node coefficients (c_i) - the points at which f(t,y) is evaluated
/// c[i] represents t_n + c[i]*h
/// 
/// From NASA TR R-287, Table X (α values)
pub const C: [f64; STAGES] = [
    0.0,          // c[0]
    2.0 / 27.0,   // c[1]  = 2/27
    1.0 / 9.0,    // c[2]  = 1/9
    1.0 / 6.0,    // c[3]  = 1/6
    5.0 / 12.0,   // c[4]  = 5/12
    0.5,          // c[5]  = 1/2
    5.0 / 6.0,    // c[6]  = 5/6
    1.0 / 6.0,    // c[7]  = 1/6
    2.0 / 3.0,    // c[8]  = 2/3
    1.0 / 3.0,    // c[9]  = 1/3
    1.0,          // c[10] = 1
    0.0,          // c[11] = 0  (for error estimation)
    1.0,          // c[12] = 1  (for error estimation)
];

/// Runge-Kutta matrix (a_ij) coefficients
/// 
/// This is the lower-triangular matrix where:
/// k_i = f(t_n + c_i*h, y_n + h * sum_{j=0}^{i-1} a_{i,j} * k_j)
///
/// Stored as A[i][j] for row i, column j (j < i)
/// From NASA TR R-287, Table X (β values)
pub const A: [[f64; 12]; 13] = [
    // Row 0: k_0 = f(t_n, y_n)
    [0.0; 12],
    
    // Row 1: k_1 = f(t_n + (2/27)*h, y_n + h*(2/27)*k_0)
    [2.0/27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 2: k_2
    [1.0/36.0, 1.0/12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 3: k_3
    [1.0/24.0, 0.0, 1.0/8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 4: k_4
    [5.0/12.0, 0.0, -25.0/16.0, 25.0/16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 5: k_5
    [1.0/20.0, 0.0, 0.0, 1.0/4.0, 1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 6: k_6
    [-25.0/108.0, 0.0, 0.0, 125.0/108.0, -65.0/27.0, 125.0/54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 7: k_7
    [31.0/300.0, 0.0, 0.0, 0.0, 61.0/225.0, -2.0/9.0, 13.0/900.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 8: k_8
    [2.0, 0.0, 0.0, -53.0/6.0, 704.0/45.0, -107.0/9.0, 67.0/90.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    
    // Row 9: k_9
    [-91.0/108.0, 0.0, 0.0, 23.0/108.0, -976.0/135.0, 311.0/54.0, -19.0/60.0, 17.0/6.0, -1.0/12.0, 0.0, 0.0, 0.0],
    
    // Row 10: k_10
    [2383.0/4100.0, 0.0, 0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0, 0.0, 0.0],
    
    // Row 11: k_11 (for 7th order error estimate)
    [3.0/205.0, 0.0, 0.0, 0.0, 0.0, -6.0/41.0, -3.0/205.0, -3.0/41.0, 3.0/41.0, 6.0/41.0, 0.0, 0.0],
    
    // Row 12: k_12 (for 7th order error estimate)
    [-1777.0/4100.0, 0.0, 0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0],
];

/// Weights for the 8th-order solution (b_i)
/// 
/// y_{n+1} = y_n + h * sum_{i=0}^{12} b[i] * k_i
///
/// Note: The 8th-order weights only use stages 0-10 (k_0 through k_10).
/// Stages 11 and 12 are only used for the 7th-order error estimate.
/// 
/// From NASA TR R-287, Table X (c values, top row)
pub const B: [f64; STAGES] = [
    41.0/840.0,     // b[0]
    0.0,            // b[1]
    0.0,            // b[2]
    0.0,            // b[3]
    0.0,            // b[4]
    34.0/105.0,     // b[5]
    9.0/35.0,       // b[6]
    9.0/35.0,       // b[7]
    9.0/280.0,      // b[8]
    9.0/280.0,      // b[9]
    41.0/840.0,     // b[10]
    0.0,            // b[11]
    0.0,            // b[12]
];

/// Weights for the 7th-order solution (b_hat_i)
/// 
/// y*_{n+1} = y_n + h * sum_{i=0}^{12} b_hat[i] * k_i
///
/// The 7th-order solution uses stages 0, 5-12.
/// 
/// From NASA TR R-287, Table X (c_hat values, bottom row)
pub const B_HAT: [f64; STAGES] = [
    0.0,            // b_hat[0]
    0.0,            // b_hat[1]
    0.0,            // b_hat[2]
    0.0,            // b_hat[3]
    0.0,            // b_hat[4]
    34.0/105.0,     // b_hat[5]
    9.0/35.0,       // b_hat[6]
    9.0/35.0,       // b_hat[7]
    9.0/280.0,      // b_hat[8]
    9.0/280.0,      // b_hat[9]
    0.0,            // b_hat[10]
    41.0/840.0,     // b_hat[11]
    41.0/840.0,     // b_hat[12]
];

/// Error weights: B[i] - B_HAT[i]
/// 
/// The local truncation error estimate is:
/// err ≈ h * sum_{i=0}^{12} (b[i] - b_hat[i]) * k_i
///
/// Note: From NASA TR R-287, the truncation error term is given as:
/// TE = (41/840) * (f_0 + f_10 - f_11 - f_12) * h
pub const B_ERR: [f64; STAGES] = [
    41.0/840.0,     // b[0] - b_hat[0]
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,            // b[5] - b_hat[5] = 0
    0.0,            // b[6] - b_hat[6] = 0
    0.0,            // b[7] - b_hat[7] = 0
    0.0,            // b[8] - b_hat[8] = 0
    0.0,            // b[9] - b_hat[9] = 0
    41.0/840.0,     // b[10] - b_hat[10]
    -41.0/840.0,    // b[11] - b_hat[11]
    -41.0/840.0,    // b[12] - b_hat[12]
];

/// Verify that the Butcher tableau satisfies the row-sum condition
/// sum_j(a_{i,j}) = c_i for all i
#[cfg(test)]
mod tests {
    use super::*;
    
    // Summation of ~13 f64 terms accumulates ~O(n*eps) roundoff
    const TOL: f64 = 1e-14;
    
    #[test]
    fn test_row_sum_condition() {
        for i in 0..STAGES {
            let row_sum: f64 = A[i].iter().sum();
            let expected = C[i];
            assert!(
                (row_sum - expected).abs() < TOL,
                "Row {} sum = {}, expected c[{}] = {}", 
                i, row_sum, i, expected
            );
        }
    }
    
    #[test]
    fn test_weights_sum_to_one() {
        let b_sum: f64 = B.iter().sum();
        assert!(
            (b_sum - 1.0).abs() < TOL,
            "8th order weights sum to {}, expected 1.0", b_sum
        );
        
        let b_hat_sum: f64 = B_HAT.iter().sum();
        assert!(
            (b_hat_sum - 1.0).abs() < TOL,
            "7th order weights sum to {}, expected 1.0", b_hat_sum
        );
    }
    
    #[test]
    fn test_error_weights_sum_to_zero() {
        let err_sum: f64 = B_ERR.iter().sum();
        assert!(
            err_sum.abs() < TOL,
            "Error weights sum to {}, expected 0.0", err_sum
        );
    }
    
    #[test]
    fn test_specific_coefficients() {
        // Verify some specific values from the table
        assert!((C[1] - 2.0/27.0).abs() < TOL);
        assert!((C[4] - 5.0/12.0).abs() < TOL);
        assert!((C[6] - 5.0/6.0).abs() < TOL);
        
        // Verify weights
        assert!((B[0] - 41.0/840.0).abs() < TOL);
        assert!((B[5] - 34.0/105.0).abs() < TOL);
        assert!((B[6] - 9.0/35.0).abs() < TOL);
    }
}
