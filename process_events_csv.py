import pandas as pd

# from IPython.display import display

mice = ["M1", "M2"]
files = ["centre_entry", "drink_start", "eat_start", "nest_entry", "social_long_start", "social_short_start"]

for mouse in mice:
    mouse_df = []
    for file_name in files:

        filepath = f"./data/MX_events/{mouse}_{file_name}_frames.csv"
        # print(f"{mouse}_{file_name}_frames.csv")
        frames = pd.read_csv(filepath, header=None).to_numpy()[0]
        # Remove consecutive frames
        # frames = [frames[0]] + [frames[idx] for idx in range(1, len(frames)) if frames[idx] - 1 != frames[idx - 1]] 
        frames = [frames[0]] + [f for i, f in enumerate(frames[1:], 1) if f != frames[i - 1] + 1]

        # print(filepath)
        # print(frames)
        # print(df)
        type = [file_name] * len(frames)
        date_time = ["01/01/2026  0:00:00 PM"] * len(frames)
        df = pd.DataFrame({"Date_Time": date_time,"Type": type, "frame": frames})
        mouse_df.append(df)
    output = mouse_df[0]
    for df in mouse_df[1:]:
        output = pd.merge(output, df, how="outer")
    output = output.sort_values('frame').reset_index(drop=True)

    output['value_changed'] = output['Type'] != output['Type'].shift(1)
    output['group_id'] = output['value_changed'].cumsum()

    grouped = output.groupby(['group_id', 'Type'], as_index=False).agg(
        group_length = ("frame", "size"),
        start_frame=("frame", "first")
    )
    eat3_df = grouped[grouped["group_length"] >= 3][grouped["Type"] == "eat_start"]
    eat3_start_frames = eat3_df["start_frame"].to_numpy()
    # print(eat3_df)
    # print(eat3_start_frames)
    output = output.drop(['value_changed', 'group_id'], axis=1)

    eat3_type = ["eat3_start"] * len(eat3_start_frames)
    eat3_date_time = ["01/01/2026  0:00:00 PM"] * len(eat3_start_frames)
    eat3_df = pd.DataFrame({"Date_Time": eat3_date_time, "Type": eat3_type, "frame": eat3_start_frames})
    # print(output)
    output = pd.merge(output, eat3_df, how="outer")
    output = output.sort_values('frame').reset_index(drop=True)
    # output = pd.merge(output, eat3_df, how="outer")

    out_path = f"./data/raw/{mouse}_events.csv"
    output.to_csv(out_path, index=False)
