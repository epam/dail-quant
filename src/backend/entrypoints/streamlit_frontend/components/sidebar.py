import streamlit as st

from market_alerts.containers import alerts_backend_proxy_singleton
from market_alerts.infrastructure.services.proxy.alerts_backend.exceptions import (
    TokenLimitsDisabled,
)


def sidebar_component():
    st.markdown("Welcome to Quantitative Research Service!\n*****************")
    try:
        token_limit = alerts_backend_proxy_singleton.get_llm_tokens_limits()
        used_tokens_amount = alerts_backend_proxy_singleton.get_used_llm_tokens_amount()
        st.session_state.llm_chat_disabled = used_tokens_amount >= token_limit
        st.markdown(f"Token usage: {used_tokens_amount:,} of {token_limit:,}\n*****************")
    except TokenLimitsDisabled:
        st.session_state.llm_chat_disabled = False
        st.markdown(f"Token limits are disabled\n*****************")
    st.markdown("\n*****************\n".join(st.session_state["log"]))
