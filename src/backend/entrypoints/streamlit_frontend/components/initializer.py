import streamlit as st

from market_alerts.entrypoints.streamlit_frontend.state import set_default_state


def initializer_component():
    set_default_state()
    st.set_page_config(page_title="Quantitative Research Service", layout="wide")

    hide_st_style = """
                    <style>

                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.markdown(
        """
      <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 500px;
        }
      </style>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
      <style>
        .move-font {
            font-size: 8px;
            visibility: hidden;
        }
        .move-logout {
            font-size: 4px;
            visibility: hidden;
        }
        .block-container {
          padding: 20px;
        }
        .streamlit-expanderHeader {
            background-color: #fffce4;
            color: black;
        }
      </style>
    """,
        unsafe_allow_html=True,
    )
