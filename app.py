import streamlit as st
import plotly.graph_objects as go
import math

def calculate_costs(P, D, nm, t_fb, t_u, b, ag, M, c_ip, c_wan, NoR, ipp):
    """
    Calculates various cost and time metrics based on input parameters.

    Args:
        P (float): No. of pipeline stages.
        D (float): No. of DP Replicas.
        nm (float): No. of microbatches processed in each iteration.
        t_fb (float): Time for forward + backward pass.
        t_u (float): Time for gradient update.
        b (float): Bandwidth.
        ag (float): Size of Activation + Gradient.
        M (float): Model Size.
        c_ip (float): Traffic cost per GB through internet.
        c_wan (float): Traffic cost per GB through WAN.
        NoR (int): No. of regions (2 or 3).
        ipp (float): Fraction of data to be transferred through internet.

    Returns:
        dict: A dictionary containing calculated metrics or an error message.
    """
    try:
        # Input validation
        if b <= 0 or D <= 0 or P <= 0 or nm <= 0:
            return {"error": "P, D, nm, and b must be greater than 0."}
        if t_fb < 0 or t_u < 0 or ag < 0 or M < 0 or c_ip < 0 or c_wan < 0 or ipp < 0 or ipp > 1:
            return {"error": "Time, size, cost, and fraction parameters cannot be negative. ipp must be between 0 and 1."}
        if NoR not in [2, 3]:
            return {"error": "NoR must be either 2 or 3."}

        # 1. t_comm (communication time)
        t_comm = ag / b

        # 2. straggler
        straggler = (nm - 1) * max(t_fb, t_comm) / D

        # 3. t_pp
        t_pp = straggler + P * t_fb + (P - 1) * t_comm + t_u

        # 4. t_sync
        t_sync = 2 * (D - 1) * M / (D * 1.45)

        # 5. t_iter
        t_iter = t_pp + t_sync

        # 6. C_comp
        C_comp = 0
        if NoR == 2:
            C_comp = t_iter * (3.73 * D * P * 0.5 + 3.81 * D * P * 0.5) / 3600.0
        elif NoR == 3:
            C_comp = t_iter * (3.73 * D * P * 0.67 + 3.81 * D * P * 0.33) / 3600.0

        # 7. C_comm_ip
        C_comm_ip = 0
        if NoR == 2:
            C_comm_ip = nm * ag * (ipp * c_ip + (1 - ipp) * c_wan)
        elif NoR == 3:
            C_comm_ip = nm * ag * (ipp * c_ip + (1 - ipp) * c_wan) * 2

        # 8. C_comm_wan
        C_comm_wan = 0
        if NoR == 2:
            C_comm_wan = nm * ag * c_wan
        elif NoR == 3:
            C_comm_wan = nm * ag * c_wan * 2

        # 9. Cost_diff
        Cost_diff = C_comm_wan - C_comm_ip

        # 10. total_cost
        total_cost = C_comp + C_comm_wan

        # 10. total_data_transferred
        total_data_transferred = 0
        if NoR == 2:
            total_data_transferred = nm * ag
        elif NoR == 3:
            total_data_transferred = nm * ag * 2

        return {
            "t_comm": t_comm,
            "straggler": straggler,
            "t_pp": t_pp,
            "t_sync": t_sync,
            "t_iter": t_iter,
            "C_comp": C_comp,
            "C_comm_ip": C_comm_ip,
            "C_comm_wan": C_comm_wan,
            "Cost_diff": Cost_diff,
            "total_cost": total_cost,
            "total_data_transferred": total_data_transferred
        }

    except Exception as e:
        return {"error": f"An error occurred during calculation: {str(e)}"}

# --- Streamlit UI ---
st.set_page_config(
    page_title="Dynamic Cost Analysis",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Dynamic Cost Analysis for Distributed Training")
st.markdown("Adjust the parameters to see their impact on communication and total costs.")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")

    P = st.number_input("P (No. of pipeline stages)", min_value=1, value=17, step=1)
    D = st.number_input("D (No. of DP Replicas)", min_value=1, value=5, step=1)
    nm = st.number_input("nm (No. of microbatches processed in each iteration)", min_value=1, value=976, step=1)
    t_fb = st.number_input("t_fb (Time for forward + backward pass)", min_value=0.0, value=0.358, step=0.001, format="%.3f")
    t_u = st.number_input("t_u (Time for gradient update)", min_value=0.0, value=0.002, step=0.001, format="%.3f")
    ag = st.number_input("ag (Size of Activation + Gradient)", min_value=0.0, value=0.0671, step=0.0001, format="%.4f")
    M = st.number_input("M (Model Size)", min_value=0.0, value=26.8, step=1.0)
    b = st.slider("b (Bandwidth)", min_value=0.1, max_value=1.0, value=0.18, step=0.01)
    c_ip = st.slider("c_ip (Traffic cost per GB through internet)", min_value=0.0, max_value=0.1, value=0.03, step=0.001)
    c_wan = st.slider("c_wan (Traffic cost per GB through WAN)", min_value=0.0, max_value=0.1, value=0.05, step=0.001)
    NoR = st.selectbox("NoR (No. of regions)", options=[2, 3], index=0)
    ipp = st.slider("ipp (Fraction of data transferred through internet)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Perform calculations
results = calculate_costs(P, D, nm, t_fb, t_u, b, ag, M, c_ip, c_wan, NoR, ipp)

if "error" in results:
    st.error(results["error"])
else:
    st.subheader("Calculated Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("t_comm (Communication Time)", f"{results['t_comm']:.4f}")
        st.metric("straggler", f"{results['straggler']:.4f}")
        st.metric("t_pp", f"{results['t_pp']:.4f}")
        st.metric("total_data_transferred", f"{results['total_data_transferred']:.4f}")
    with col2:
        st.metric("t_sync", f"{results['t_sync']:.4f}")
        st.metric("t_iter", f"{results['t_iter']:.4f}")
        st.metric("C_comp (Computation Cost)", f"${results['C_comp']:.2f}")
    with col3:
        st.metric("C_comm_ip (Internet Comm. Cost)", f"${results['C_comm_ip']:.2f}")
        st.metric("C_comm_wan (WAN Comm. Cost)", f"${results['C_comm_wan']:.2f}")
        st.metric("Cost_diff (WAN - Internet)", f"${results['Cost_diff']:.2f}")
        st.metric("Total Cost (C_comp + C_comm_wan)", f"${results['total_cost']:.2f}")

    st.markdown("---")
    st.subheader("Cost Difference vs. Total Cost")

    # Prepare data for Plotly pie chart
    total_cost = results['total_cost']
    cost_diff = results['Cost_diff']

    if total_cost <= 1e-9: # Handle near-zero total_cost
        st.warning("Total cost is too small or zero to generate a meaningful pie chart. Please adjust inputs.")
    else:
        # For the pie chart, we need non-negative values that sum up to a whole.
        # We want to show "how small 'cost_diff' is with respect to 'total_cost'".
        # If Cost_diff is positive, it's a cost component.
        # If Cost_diff is negative, it's a saving (C_comm_ip is cheaper than C_comm_wan).
        # To represent this consistently in a pie chart where slices are parts of a whole:
        # We'll use max(0, cost_diff) for the "Positive Cost Difference" slice.
        # The "Remaining Total Cost" slice will be total_cost - max(0, cost_diff).
        # This ensures both slices are non-negative and sum to total_cost.

        positive_cost_difference_slice = max(0, cost_diff)
        remaining_total_cost_slice = total_cost - positive_cost_difference_slice

        labels = ['Positive Cost Difference (WAN - Internet)', 'Total Cost (excluding positive difference)']
        values = [positive_cost_difference_slice, remaining_total_cost_slice]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                     hoverinfo='label+percent+value',
                                     textinfo='percent',
                                     marker=dict(colors=['#ef4444', '#3b82f6']), # Red for Cost Diff, Blue for Remaining
                                     pull=[0.05, 0] # Pull out the Cost Difference slice slightly
                                     )])

        fig.update_layout(
            title_text="Proportion of Positive Cost Difference in Total Cost",
            title_font_size=20,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=100, b=50, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        if cost_diff < 0:
            st.info(f"Note: The 'Cost Difference' is negative (${abs(cost_diff):.2f}), indicating a saving when using internet traffic compared to WAN traffic for the fraction specified by `ipp`.")
