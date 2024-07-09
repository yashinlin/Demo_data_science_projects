def bar_x_hue_p(df, x, hue, title):
    sns.set_style("whitegrid")

    df_p = (
        df.groupby(x)[hue]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
        .reset_index()  # reset_index needed bc otherwise index of original Series will remain
    )

    sns.catplot(
        data=df_p,
        x=x,
        y="percent",
        hue=hue,
        palette=["#CD2E3A", "#0047A0", "#6600cc"],
        order=[
            "0s",
            "10s",
            "20s",
            "30s",
            "40s",
            "50s",
            "60s",
            "70s",
            "80s",
            "90s",
            "100s",
            "missing",
        ],
        kind="bar",
    ).set(title=title)

# 3 barcharts ==============================================================
def bar_age_sex_by_status(df_by_state):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 6),
    )

    for i, status in enumerate(["isolated", "released", "deceased"]):
        # create df from dictionary
        df = pd.DataFrame(df_by_state[status]) 

        # create variable that counts total by age and sex
        ct = df.groupby("age")["sex"].value_counts().rename("count").reset_index()

        sns.barplot(
            data=ct,
            x="age",
            y="count",
            hue="sex",
            order=[
                "0s",
                "10s",
                "20s",
                "30s",
                "40s",
                "50s",
                "60s",
                "70s",
                "80s",
                "90s",
                "100s",
                "missing",
            ],
            ax=axes[i],
            palette=["#CD2E3A", "#0047A0", "#D3D3D3"],
        ).set(title=f"{status.capitalize()} cases")
        if i != 0:
            axes[i].legend_.remove()
        else:
            axes[i].legend(loc=(0.65, 0.8))

    fig.suptitle(
        "Figure 5. Distribution of Covid isolated, released and deceased cases, by age and sex. Jan-Jun 2020 \n(Daegu excluded; Source: local government sites)",
        fontsize=14,
        y=1.05,
    )
   
