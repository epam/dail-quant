from market_alerts.domain.constants import (
    ADDITIONAL_COLUMNS,
    COLUMNS_TO_DESCRIPTION,
    INDICATORS,
    KEYS_TO_COLUMNS,
)


def define_useful_strings(session_dict):
    session_dict["u_strs"] = dict()

    all_indicators_str = "\n".join([key + "," + INDICATORS[key] for key in INDICATORS])
    input_securities_str = "\n".join([key + ", " + value for key, value in session_dict["meta_by_symbol"].items()])
    list_securities_str = ", ".join(session_dict["symbols"] + session_dict["synths"])
    list_securities_withv_str = ", ".join(
        [i for i in session_dict["symbols"] if "volume" in session_dict["data_by_symbol"].get(i, {})]
        + [i for i in session_dict["synths"] if "volume" in session_dict["data_by_synth"].get(i, {})]
    )
    list_securities_withoutv_str = ", ".join(
        [i for i in session_dict["symbols"] if not "volume" in session_dict["data_by_symbol"].get(i, {})]
        + [i for i in session_dict["synths"] if not "volume" in session_dict["data_by_synth"].get(i, {})]
    )
    list_sparse_securities_str = ", ".join(session_dict["sparse_symbols"])
    list_economic_indicator_symbols_str = ", ".join(session_dict["economic_indicator_symbols"])
    columns = []
    for key in session_dict["datasets_keys"]:
        if key in KEYS_TO_COLUMNS:
            columns.append(KEYS_TO_COLUMNS[key])

    for key in session_dict["dividend_fields"]:
        if key in KEYS_TO_COLUMNS:
            columns.append(KEYS_TO_COLUMNS[key])
    additional_columns_table = "\n".join(["- %s: %s" % (col, COLUMNS_TO_DESCRIPTION[col]) for col in columns])

    session_dict["u_strs"]["all_indicators_str"] = all_indicators_str
    session_dict["u_strs"]["input_securities_str"] = input_securities_str
    session_dict["u_strs"]["list_securities_str"] = list_securities_str
    session_dict["u_strs"]["list_securities_withv_str"] = list_securities_withv_str
    session_dict["u_strs"]["list_securities_withoutv_str"] = list_securities_withoutv_str
    session_dict["u_strs"]["list_sparse_securities_str"] = list_sparse_securities_str
    session_dict["u_strs"]["list_economic_indicator_symbols_str"] = list_economic_indicator_symbols_str
    session_dict["u_strs"]["additional_columns_table"] = additional_columns_table


def define_empty_indicators_step(session_dict):
    symbols_and_synths = session_dict["symbols"] + session_dict["synths"]
    symbols_and_synth_data = dict()
    for key in ["data_by_symbol", "data_by_synth"]:
        symbols_and_synth_data.update(session_dict[key])
    roots = dict()
    for s in symbols_and_synths:
        roots[s] = []
        for a_column in ADDITIONAL_COLUMNS:
            if a_column in symbols_and_synth_data.get(s, {}):
                roots["%s.%s" % (s, a_column)] = []
    main_roots = dict()
    for s in symbols_and_synths:
        main_roots[s] = []
        for a_column in ADDITIONAL_COLUMNS:
            if a_column in symbols_and_synth_data.get(s, {}):
                column_id = "%s.%s" % (s, a_column)
                main_roots[s].append(column_id)
    session_dict["roots"] = roots
    session_dict["main_roots"] = main_roots
    session_dict["indicators"] = []
    session_dict["data_by_indicator"] = dict()
    session_dict["u_strs"]["indicators_str"] = ""
    session_dict["indicators_code"] = ""
