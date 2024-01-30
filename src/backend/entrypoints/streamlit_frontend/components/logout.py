import streamlit as st

from market_alerts.entrypoints.streamlit_frontend.state import logout_callback


def logout_component(email_phold, logout_button_phold):
    email_phold.text("\n")
    email_phold.text("\n")
    email_phold.text(st.session_state.keycloak_session.user_info["email"])

    logout_button_phold.markdown('<p class="move-logout">T</p>', unsafe_allow_html=True)
    if logout_button_phold.button(
        "Sign out",
        use_container_width=True,
        on_click=logout_callback,
        args=[st.session_state.keycloak_session.id_token],
    ):
        st.experimental_rerun()
