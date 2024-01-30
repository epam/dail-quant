import os
import sys

import streamlit as st
from streamlit_keycloak import login

from market_alerts.config import KEYCLOAK_CLIENT_ID
from market_alerts.containers import alerts_backend_proxy_singleton

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from market_alerts.entrypoints.streamlit_frontend.components import (
    initializer_component,
    logout_component,
    sidebar_component,
)
from market_alerts.entrypoints.streamlit_frontend.components.tabs import (
    main_tab_component,
    public_models_component,
    trading_tab_component,
)
from market_alerts.openai_utils import *


def main():
    st.elements.utils._shown_default_value_warning = True
    initializer_component()

    col1, _, col3, col4, col5 = st.columns([0.07, 0.15, 0.56, 0.15, 0.07])
    reset_phold = col1.container()
    header_phold = col3.empty()
    email_phold = col4.container()
    logout_button_phold = col5.container()

    header_phold.markdown(
        "<h2 style='text-align: center; color: black; '>Quantitative Research Service</h2>",
        unsafe_allow_html=True,
    )

    if st.session_state.logged_out:
        st.markdown(
            "<h1 style='text-align: center; color: black; '>You have signed out</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h2 style='text-align: center; color: black; '>Please reload the page</h2>",
            unsafe_allow_html=True,
        )
        st.stop()

    col1, col2, col3 = st.columns([0.35, 0.3, 0.35])
    login_phold = col2.empty()

    with login_phold:
        st.session_state.keycloak_session = login(
            url="https://kc.staging.deltixhub.io",
            realm="market_alerts",
            client_id=KEYCLOAK_CLIENT_ID,
        )

    if st.session_state.keycloak_session.authenticated:
        auth_header = {"Authorization": f"Bearer {st.session_state.keycloak_session.access_token}"}
        alerts_backend_proxy_singleton.set_auth_header(auth_header)
        alerts_backend_proxy_singleton.set_email(st.session_state.keycloak_session.user_info["email"])
    else:
        st.stop()

    main_tab, trading_tab, public_trading_tab = st.tabs(["Design", "My Models", "Public Models"])

    css = """
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
        }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    st.markdown(
        """
            <style>
                button.step-up {display: none;}
                button.step-down {display: none;}
                div[data-baseweb] {border-radius: 4px;}
            </style>""",
        unsafe_allow_html=True,
    )

    reset_phold.markdown('<p class="move-logout">T</p>', unsafe_allow_html=True)
    if reset_phold.button("Reset", on_click=st.session_state.clear):
        st.experimental_rerun()

    if st.session_state.keycloak_session is not None and st.session_state.keycloak_session.authenticated:
        logout_component(email_phold, logout_button_phold)

    with st.sidebar:
        sidebar_component()

    with main_tab:
        main_tab_component()

    with trading_tab:
        trading_tab_component()

    with public_trading_tab:
        public_models_component()


if __name__ == "__main__":
    main()
