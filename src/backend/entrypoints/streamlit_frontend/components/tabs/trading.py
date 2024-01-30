import json
from datetime import datetime

import requests
import streamlit as st

from market_alerts.config import ALERTS_BACKEND_SERVICE_URL
from market_alerts.entrypoints.streamlit_frontend.state import (
    paginate,
    trading_delete_callback,
    trading_rule_edit_callback,
)


def trading_tab_component():
    headers = {"Authorization": "Bearer " + st.session_state.keycloak_session.access_token}

    resp = requests.get(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/trading-info", headers=headers)

    tradings = json.loads(resp.content)

    columns_titles = [
        "**Title**",
        "**Date**",
        "**Shared by**",
        "**Tickers**",
        "",
    ]
    column_width = [0.2, 0.18, 0.18, 0.3, 0.07, 0.07]
    title_cols = st.columns(column_width, gap="small")

    for index, title in enumerate(columns_titles):
        column_phold = title_cols[index].empty()
        column_phold.markdown(title)

    reset_phold = title_cols[-1].empty()
    reset_phold.button("Refresh", use_container_width=True, key="trading_models_refresh")

    def show_tradings(tradings):
        for trading in tradings:
            description = trading.get("description", "") if trading.get("description", "") is not None else ""

            html_code = f"""
                <style> 
                .container {{ 
                    display: flex; 
                    align-items: center; 
                }} 
                .text {{ 
                    margin-right: 10px; 
                }} 
                </style> 
                <div class="container"> 
                    <span class="text">{trading["title"]}</span> 
                    <img src="app/static/QuestionMark.png" title="{description}" width="15" height="15">
                </div>
            """
            (
                title_col,
                date_col,
                shared_by_col,
                tickers_col,
                edit_col,
                delete_col,
            ) = st.columns(column_width, gap="small")

            title_phold = title_col.empty()
            title_phold.write(html_code, unsafe_allow_html=True)

            tickers_phold = tickers_col.empty()
            tickers_phold.write(trading["tickers_prompt"].replace(":", "\:"))

            date_phold = date_col.empty()
            date = datetime.strptime(trading["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            trimmed_time = date.strftime("%Y-%m-%d %H:%M")
            date_phold.write(trimmed_time)

            shared_by_phold = shared_by_col.empty()

            if st.session_state.keycloak_session.user_info["email"] == trading["author_id"]:
                trading["author_id"] = ""

            shared_by_phold.write(trading["author_id"] if trading["author_id"] is not None else "")

            edit_phold = edit_col.empty()
            if edit_phold.button(
                "Open",
                key="edit_btn_" + str(trading["id"]),
                use_container_width=True,
                on_click=trading_rule_edit_callback,
                args=[trading["id"], st.session_state.keycloak_session.access_token],
            ):
                st.experimental_rerun()

            delete_phold = delete_col.empty()
            if delete_phold.button(
                "Delete",
                key="delete_btn_" + str(trading["id"]),
                use_container_width=True,
            ):
                warning_col, submit_col, cancel_col = st.columns([0.8, 0.1, 0.1])
                warning_col_phold = warning_col.container()
                submit_phold = submit_col.container()
                cancel_phold = cancel_col.container()
                warning_col_phold.warning(f"Are you sure you want to delete model {trading['title']} ?")

                submit_phold.button(
                    "Confirm",
                    use_container_width=True,
                    on_click=trading_delete_callback,
                    args=[trading["id"], st.session_state.keycloak_session.access_token],
                )

                cancel_phold.button("Cancel", use_container_width=True)

    paginate(
        data=tradings,
        show_func=show_tradings,
        page_num=st.session_state.trading_page_num if len(tradings) else 0,
        page_num_key="trading_page_num",
        page_size=st.session_state.trading_page_size if len(tradings) else 10,
        page_size_key="trading_page_size",
    )
