import streamlit as st

from market_alerts.entrypoints.streamlit_frontend.state import (
    event_list_update_callback,
    paginate,
)


def events_tab_component():
    column_names = [
        "**Alert title**",
        "**Event creation time**",
        "**Event status**",
        "**Event message**",
    ]
    columns_width = [0.1, 0.2, 0.1, 0.5, 0.1]

    cols = st.columns(columns_width, gap="small")

    for index, col_name in enumerate(column_names):
        title_phold = cols[index].container()
        title_phold.markdown(col_name)

    update_btn_phold = cols[index + 1].container()
    update_btn_phold.button("Refresh", use_container_width=True, on_click=event_list_update_callback)

    if len(st.session_state.events) == 0:
        event_list_update_callback()

    def show_events(events):
        for event in events:
            event_alert_title_col, event_created_col, event_status_col, event_message_col = st.columns([0.1, 0.2, 0.1, 0.6])
            event_alert_title_phold = event_alert_title_col.container()
            event_alert_title_phold.write(event["alert_title"])

            time_creation = " ".join(event["timestamp"].split("T")).split("Z")[0]
            event_created_phold = event_created_col.container()
            event_created_phold.write(time_creation)

            event_status_phold = event_status_col.container()
            event_status_phold.write(event["status"])

            event_message_phold = event_message_col.container()
            event_message_phold.text(event["message"])

    paginate(
        data=st.session_state.events,
        show_func=show_events,
        page_num=st.session_state.event_page_num if len(st.session_state.events) else 0,
        page_num_key="event_page_num",
        page_size=st.session_state.event_page_size if len(st.session_state.events) else 10,
        page_size_key="event_page_size",
    )
