from datetime import datetime

import streamlit as st

from market_alerts.containers import alerts_backend_proxy_singleton
from market_alerts.entrypoints.streamlit_frontend.state import (
    paginate,
    trading_delete_callback,
    trading_rule_edit_callback,
)


def public_models_component():
    public_tradings = alerts_backend_proxy_singleton.get_public_trading_rules()

    if st.session_state["is_admin"]:
        columns_titles = [
            "**Title**",
            "**Date**",
            "**Shared by**",
            "**Tickers**",
            "",
        ]
        column_width = [0.2, 0.18, 0.18, 0.3, 0.07, 0.07]
    else:
        columns_titles = [
            "**Title**",
            "**Date**",
            "**Shared by**",
            "**Tickers**",
        ]
        column_width = [0.2, 0.18, 0.18, 0.3, 0.14]

    title_cols = st.columns(column_width, gap="small")

    for index, title in enumerate(columns_titles):
        column_phold = title_cols[index].empty()
        column_phold.markdown(title)

    reset_phold = title_cols[-1].empty()
    reset_phold.button("Refresh", use_container_width=True, key="public_models_refresh")

    def show_public_tradings(tradings):
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

            cols = st.columns(column_width, gap="small")

            title_phold = cols[0].empty()
            # title_phold.text(trading["title"], help=description)
            title_phold.write(html_code, unsafe_allow_html=True)

            date_phold = cols[1].empty()
            date = datetime.strptime(trading["created"], "%Y-%m-%dT%H:%M:%S.%fZ")
            trimmed_time = date.strftime("%Y-%m-%d %H:%M")
            date_phold.write(trimmed_time)

            shared_by_phold = cols[2].empty()

            shared_by_phold.write(trading["author_id"] if trading["author_id"] is not None else "")

            tickers_phold = cols[3].empty()
            tickers_phold.write(trading["tickers_prompt"].replace(":", "\:"))

            edit_phold = cols[4].empty()
            if edit_phold.button(
                "Open",
                key="publ_edit_btn_" + str(trading["id"]),
                use_container_width=True,
                on_click=trading_rule_edit_callback,
                args=[trading["id"], st.session_state.keycloak_session.access_token, True],
            ):
                st.experimental_rerun()

            if st.session_state["is_admin"]:
                delete_phold = cols[5].empty()
                if delete_phold.button(
                    "Delete",
                    key="publ_delete_btn_" + str(trading["id"]),
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
                        args=[trading["id"], st.session_state.keycloak_session.access_token, True],
                    )

                    cancel_phold.button("Cancel", use_container_width=True)

    paginate(
        data=public_tradings,
        show_func=show_public_tradings,
        page_num=st.session_state.public_page_num if len(public_tradings) else 0,
        page_num_key="public_page_num",
        page_size=st.session_state.public_page_size if len(public_tradings) else 10,
        page_size_key="public_page_size",
    )
