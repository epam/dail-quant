import json

import requests
import streamlit as st

from market_alerts.config import ALERTS_BACKEND_SERVICE_URL
from market_alerts.domain.constants import PROMPTS_SEPARATOR
from market_alerts.entrypoints.streamlit_frontend.state import (
    alert_active_callback,
    alert_delete_callback,
    alert_edit_callback,
    paginate,
)


def alerts_tab_component():
    headers = {"Authorization": "Bearer " + st.session_state.keycloak_session.access_token}

    resp = requests.get(f"{ALERTS_BACKEND_SERVICE_URL}/api/v0/alerts", headers=headers)

    alerts = json.loads(resp.content)

    columns_titles = [
        "**Title**",
        "**Active**",
        "**Tickers**",
        "**Alert prompt**",
        "**Trigger type**",
        "**Last call**",
        "**Creation time**",
        "",
        "",
    ]
    column_width = [0.09, 0.07, 0.17, 0.17, 0.07, 0.12, 0.12, 0.07, 0.07]
    title_cols = st.columns(column_width, gap="small")

    for index, column in enumerate(title_cols):
        column_phold = column.empty()
        column_phold.markdown(columns_titles[index])

    def show_alerts(alerts):
        for alert in alerts:
            (
                alert_title_col,
                active_col,
                tickers_col,
                alerts_pr_col,
                trigger_tp_col,
                last_call_col,
                created_col,
                edit_col,
                delete_col,
            ) = st.columns(column_width, gap="small")

            alert_title_phold = alert_title_col.empty()
            alert_title_phold.write(alert["title"])

            active_phold = active_col.empty()
            active_phold.checkbox(
                "Active",
                value=alert["active"],
                key="active_check_" + str(alert["id"]),
                label_visibility="hidden",
                on_change=alert_active_callback,
                args=[
                    alert,
                    st.session_state.keycloak_session.access_token,
                    "active_check_" + str(alert["id"]),
                ],
            )

            tickers_phold = tickers_col.empty()
            tickers_phold.write(alert["tickers_prompt"])

            alerts_pr_phold = alerts_pr_col.empty()
            alert_prompts = alert["alerts_prompt"].split(PROMPTS_SEPARATOR)[::2]
            alerts_pr_phold.write("\n\n".join(alert_prompts))

            trigger_tp_phold = trigger_tp_col.empty()
            trigger_tp_phold.write(alert["trigger_type"])

            last_call_phold = last_call_col.empty()

            if alert["alert_state"] is not None and alert["alert_state"]["last_event_time"] is not None:
                last_call = " ".join(alert["alert_state"]["last_event_time"].split("T"))
                last_call = last_call.split("Z")[0]
            else:
                last_call = "Hasn't been called yet"

            last_call_phold.write(last_call)

            created = " ".join(alert["created"].split("T"))
            created = created.split("Z")[0]
            created_phold = created_col.empty()
            created_phold.write(created)

            edit_phold = edit_col.empty()
            if edit_phold.button(
                "Edit",
                key="edit_btn_" + str(alert["id"]),
                use_container_width=True,
                on_click=alert_edit_callback,
                args=[alert["id"], st.session_state.keycloak_session.access_token],
            ):
                st.experimental_rerun()

            delete_phold = delete_col.empty()
            if delete_phold.button(
                "Delete",
                key="delete_btn_" + str(alert["id"]),
                use_container_width=True,
            ):
                warning_col, submit_col, cancel_col = st.columns([0.8, 0.1, 0.1])
                warning_col_phold = warning_col.container()
                submit_phold = submit_col.container()
                cancel_phold = cancel_col.container()
                warning_col_phold.warning(f"Are you sure you want to delete alert {alert['title']} ?")

                submit_phold.button(
                    "Confirm",
                    use_container_width=True,
                    on_click=alert_delete_callback,
                    args=[alert["id"], st.session_state.keycloak_session.access_token],
                )

                cancel_phold.button("Cancel", use_container_width=True)

    paginate(
        data=alerts,
        show_func=show_alerts,
        page_num=st.session_state.alert_page_num if len(alerts) else 0,
        page_num_key="alert_page_num",
        page_size=st.session_state.alert_page_size if len(alerts) else 10,
        page_size_key="alert_page_size",
    )
