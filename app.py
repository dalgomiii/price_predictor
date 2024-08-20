import streamlit as st
import numpy as np
from scipy.stats import norm

def black_scholes_european_call(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_european_put(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def binomial_american_call(S, X, T, r, sigma, max_steps, t):
    dt = T / max_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(max_steps + 1)
    option_values = np.zeros(max_steps + 1)
    
    for i in range(max_steps + 1):
        asset_prices[i] = S * (u ** (max_steps - i)) * (d ** i)
    
    # Calculate option values at maturity
    for i in range(max_steps + 1):
        option_values[i] = max(0, asset_prices[i] - X)
    
    # Backward induction up to t steps from maturity
    for j in range(max_steps - 1, max_steps - t - 1, -1):
        for i in range(j + 1):
            option_values[i] = (p * option_values[i] + (1 - p) * option_values[i + 1]) * np.exp(-r * dt)
            asset_prices[i] = asset_prices[i] / u
            option_values[i] = max(option_values[i], asset_prices[i] - X)
    
    return option_values[0]

def binomial_american_put(S, X, T, r, sigma, max_steps, t):
    dt = T / max_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(max_steps + 1)
    option_values = np.zeros(max_steps + 1)
    
    for i in range(max_steps + 1):
        asset_prices[i] = S * (u ** (max_steps - i)) * (d ** i)
    
    # Calculate option values at maturity
    for i in range(max_steps + 1):
        option_values[i] = max(0, X - asset_prices[i])
    
    # Backward induction up to t steps from maturity
    for j in range(max_steps - 1, max_steps - t - 1, -1):
        for i in range(j + 1):
            option_values[i] = (p * option_values[i] + (1 - p) * option_values[i + 1]) * np.exp(-r * dt)
            asset_prices[i] = asset_prices[i] / u
            option_values[i] = max(option_values[i], X - asset_prices[i])
    
    return option_values[0]

# Streamlit app
st.title("Option Pricing Calculator")

st.sidebar.header("Input Parameters")

# Option type selection
option_type = st.sidebar.selectbox("Option Type", ("European", "American"))

# Default values
default_values = {
    "S": 100.0,
    "X": 100.0,
    "T": 1.0,
    "r": 0.05,
    "sigma": 0.2,
    "max_steps": 100,
    "steps": 100
}

# Create input fields in the sidebar
S = st.sidebar.number_input("Stock Price (S)", value=default_values["S"], min_value=0.0, step=1.0, format="%.2f",max_value=1000000.0)
X = st.sidebar.number_input("Strike Price (X)", value=default_values["X"], min_value=0.0, step=1.0, format="%.2f", max_value=1000000.0)
T = st.sidebar.number_input("Time to Expiry (T in years)", value=default_values["T"], min_value=0.0, step=0.1, format="%.2f", max_value=10.0)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=default_values["r"], min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=default_values["sigma"], min_value=0.0, max_value=1.0, step=0.01, format="%.4f")

# Conditional inputs for American options
if option_type == "American":
    max_steps = st.sidebar.number_input("Maximum Time Steps (n)", value=default_values["max_steps"], min_value=1, step=1, max_value=3650)
    col1, col2 = st.sidebar.columns(2)
    steps = col1.slider("Time Steps (t)", min_value=0, max_value=max_steps, value=max_steps)
    steps_input = col2.number_input("",value=steps, min_value=0, max_value=max_steps, step=1)
    steps = steps_input

# Calculate option prices based on option type
if option_type == "European":
    call_price = black_scholes_european_call(S, X, T, r, sigma)
    put_price = black_scholes_european_put(S, X, T, r, sigma)
    # Display the Black-Scholes formulas
    st.markdown("### Black-Scholes Formula for European Options\n where $C$ is the call option price, $P$ is the put option price, $N(d)$ is the cumulative distribution function of the standard normal distribution")
    st.latex(r"C = S \cdot N(d_1) - X \cdot e^{-rT} \cdot N(d_2)")
    st.latex(r"P = X \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)")
    st.latex(r"d_1 = \frac{\ln\left(\frac{S}{X}\right) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}")
    st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")
else:
    call_price = binomial_american_call(S, X, T, r, sigma, max_steps, steps)
    put_price = binomial_american_put(S, X, T, r, sigma, max_steps, steps)
    # Display the Binomial model description
    st.markdown("### Binomial Model for American Options")
    # Putting the formulas in LaTeX format in a container
    with st.container(height=300):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**Up factor**, $u$")
            st.latex(r"u = e^{\sigma \sqrt{\Delta t}}")
        with col2:
            st.markdown("**Down factor**, $d$")
            st.latex(r"d = \frac{1}{u}")
        with col3:
            st.markdown("**Risk-neutral probability**, $p$")
            st.latex(r"p = \frac{e^{r \Delta t} - d}{u - d}")
        with col4:
            st.markdown("Time Step, $\Delta t$")
            st.latex(r"\Delta t = \frac{T}{n}")
        st.markdown("#### Option Price Calculation at Specific Time Step \n where $S_t$ is the price of the stock at $t$ time steps, $n_u$ is the number of up movements and $n_d$ is the number of down movements")
        st.latex(r"S_t = S_0 \times u^{n_u} \times d^{n_d}")
        st.markdown("#### Intrinsic Value and Discounted Expected Value for Call and Put Options \n where $C_u$ is the call option price at $t+1$ time steps if the stock price goes up, $C_d$ is the call option price at $t+1$ time steps if the stock price goes down, $P_u$ is the put option price at $t+1$ time steps if the stock price goes up, and $P_d$ is the put option price at $t+1$ time steps if the stock price goes down")
        st.markdown(r'''
| **Option Type**   | **Intrinsic Value**                                          | **Discounted Expected Value**                                                                 |
|-------------------|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **American Call** |  $\max(S_t - X, 0)$                            |  $ e^{-r\Delta t} \times (pC_u + (1-p)C_d)$                                    |
| **American Put**  | $\max(X - S_t, 0)$                        | $e^{-r\Delta t} \times (pP_u + (1-p)P_d)$                                   |
''')
        st.markdown("#### Value of Option Adjust for Early Exercise")
        st.latex(r"V_t = \max(\text{Intrinsic Value}, \text{Discounted Expected Value})")

# Display the results
st.write(f"### {option_type} Option Prices")
st.write(f"**Call Option Price**: ${call_price:.2f}")
st.write(f"**Put Option Price**: ${put_price:.2f}")

