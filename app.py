import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Colour palette for charts
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

@st.cache_data
def load_data():
    return pd.read_csv("atelier8_survey_data.csv")


def main():

    st.set_page_config(
        page_title="ATELIER 8 – Circular Luxury Intelligence Dashboard",
        page_icon="👜",
        layout="wide"
    )

    # ---------- UPDATED BACKGROUND + HEADER + TEAM BADGE ----------
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FFF4E2; /* Warm beige luxury background */
        }

        h1 {
            color: #D4AF37 !important; /* Luxury golden heading */
        }

        .team-bar {
            position: absolute;
            top: 18px;
            right: 30px;
            background-color: #FFE9C7;
            padding: 6px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            color: #A67800;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
            white-space: nowrap;
            animation: slide 10s linear infinite;
        }

        @keyframes slide {
            0% { transform: translateX(25%); }
            50% { transform: translateX(-25%); }
            100% { transform: translateX(25%); }
        }
        </style>

        <div class="team-bar">
            Kanav | Jigyasa | Omkar | Hardik | Harshal
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- HEADER ----------
    st.markdown(
        "<h1 style='margin-bottom:0px;'>ATELIER 8 – Circular Luxury Intelligence Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#555; font-size:15px; margin-top:4px;'>"
        "Data-driven insights for restoration, authentication, and circular luxury in the UAE."
        "</p>",
        unsafe_allow_html=True,
    )

    # ---------- INTRO EXPANDER ----------
    with st.expander("📌 What is this dashboard about?", expanded=True):
        st.write(
            """
            **Business idea**  
            ATELIER 8 is a circular luxury restoration & authentication studio for handbags and sneakers.

            **Dataset**  
            Synthetic survey of **400 potential UAE luxury consumers** – age, income, luxury ownership,
            sustainability importance, resale interest and willingness-to-pay (WTP).

            **Dashboard goal**  
            Help a **non-technical viewer** understand:
            - Who the potential customers are  
            - How they can be segmented into simple personas  
            - How much they are willing to pay  
            """
        )

    df = load_data()

    # ---------- SIDEBAR FILTERS ----------
    st.sidebar.title("🎛 Filters")
    st.sidebar.write("Slice the dashboard for specific audience pockets.")

    age_filter = st.sidebar.multiselect(
        "Age group", sorted(df["age_group"].unique()), sorted(df["age_group"].unique())
    )
    income_filter = st.sidebar.multiselect(
        "Income level", sorted(df["income_level"].unique()), sorted(df["income_level"].unique())
    )
    gender_filter = st.sidebar.multiselect(
        "Gender", sorted(df["gender"].unique()), sorted(df["gender"].unique())
    )

    filtered_df = df[
        df["age_group"].isin(age_filter)
        & df["income_level"].isin(income_filter)
        & df["gender"].isin(gender_filter)
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Current sample: **{len(filtered_df)} respondents**")

    # ---------- TABS ----------
    tab_overview, tab_segments, tab_models, tab_upload = st.tabs(
        ["1️⃣ Overview & Story", "2️⃣ Customer Segments", "3️⃣ Predictive Models", "4️⃣ Upload & Score New Data"]
    )

    # ---------- TAB 1: OVERVIEW ----------
    with tab_overview:
        st.subheader("1️⃣ Business Overview – Who are ATELIER 8’s potential customers?")

        st.info(
            "This section answers: Who did we survey? How ready are they to try ATELIER 8? "
            "How much are they willing to spend?"
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total respondents", len(filtered_df))
        with col2:
            st.metric("% likely to adopt (adoption ≥ 4)", f"{(filtered_df['adoption_intent']>=4).mean()*100:.1f}%")
        with col3:
            st.metric("Avg WTP – restoration", f"AED {filtered_df['wtp_restoration_aed'].mean():,.0f}")
        with col4:
            st.metric("Avg WTP – authentication", f"AED {filtered_df['wtp_authentication_aed'].mean():,.0f}")

        # Pie Charts
        st.markdown("### A. Audience mix at a glance")
        colA, colB = st.columns(2)

        with colA:
            fig = px.pie(
                filtered_df, names="income_level",
                title="Customer split by income level",
                color="income_level", color_discrete_sequence=PALETTE
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig = px.pie(
                filtered_df, names="gender",
                title="Customer split by gender",
                color="gender", color_discrete_sequence=PALETTE
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        # Adoption & sustainability
        st.markdown("### B. Adoption & sustainability view")
        colC, colD = st.columns(2)

        with colC:
            adoption_age = (
                filtered_df.groupby("age_group")["adoption_intent"]
                .mean().reset_index().sort_values("adoption_intent", ascending=False)
            )
            fig = px.bar(
                adoption_age,
                x="age_group", y="adoption_intent",
                color="age_group", color_discrete_sequence=PALETTE,
                title="Average adoption intention by age group"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with colD:
            fig = px.histogram(
                filtered_df, x="sustainability_importance", nbins=5,
                title="Sustainability importance distribution",
                color_discrete_sequence=[PALETTE[2]]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Ownership & WTP
        st.markdown("### C. Luxury ownership & willingness to pay")
        colE, colF = st.columns(2)

        with colE:
            brands = ["owns_chanel", "owns_lv", "owns_dior", "owns_gucci", "owns_hermes", "owns_sneaker_grails"]
            bc = filtered_df[brands].sum().reset_index()
            bc.columns = ["brand", "count"]

            fig = px.bar(
                bc, x="brand", y="count",
                title="Ownership of key luxury brands",
                color="brand", color_discrete_sequence=PALETTE
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with colF:
            fig = px.box(
                filtered_df,
                x="income_level", y="wtp_restoration_aed",
                title="WTP for restoration by income level",
                color="income_level", color_discrete_sequence=PALETTE
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### D. Sustainability ↔ WTP relationship")
        fig = px.scatter(
            filtered_df,
            x="sustainability_importance", y="wtp_restoration_aed",
            color="income_level", color_discrete_sequence=PALETTE,
            title="Sustainability vs WTP (restoration)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 2: SEGMENTS ----------
    with tab_segments:
        st.subheader("2️⃣ Customer Segments – Data-driven personas")

        st.info("We use **K-Means Clustering** to discover patterns and natural customer groups.")

        st.markdown(
            """
            **What is clustering and K-Means, and why are we using it?**  
            - *Clustering* groups similar customers together without any pre-defined labels – it lets the data reveal natural segments.  
            - **K-Means** is a simple, widely used clustering algorithm that assigns every customer to one of *K* groups based on similarity in behaviour.  
            - For ATELIER 8, this helps us turn raw survey responses into **clear personas** (e.g., high-income collectors, sneaker heads, conscious curators) for targeted strategies.
            """
        )

        features = filtered_df[
            ["wtp_restoration_aed", "wtp_authentication_aed",
             "sustainability_importance", "handbags_owned",
             "sneakers_owned", "resale_interest"]
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        k = st.slider("Choose number of clusters (K)", 2, 6, 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        filtered_df["cluster"] = kmeans.fit_predict(X_scaled)

        col1, col2 = st.columns([1.3, 1])

        with col1:
            fig = px.scatter(
                filtered_df,
                x="wtp_restoration_aed", y="wtp_authentication_aed",
                color="cluster", color_discrete_sequence=PALETTE,
                hover_data=["income_level", "age_group", "sustainability_importance", "resale_interest"],
                title="Customer segments by WTP (restoration vs authentication)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            summary = (
                filtered_df.groupby("cluster")[
                    ["wtp_restoration_aed", "wtp_authentication_aed",
                     "sustainability_importance", "handbags_owned",
                     "sneakers_owned", "resale_interest"]
                ]
                .mean()
                .round(1)
            )
            st.markdown("**Cluster profile summary (averages)**")
            st.dataframe(summary)

        # ---- Persona chips + downloadable summary ----
        available_clusters = sorted(summary.index.tolist())

        persona_labels = {
            0: "🟣 **Cluster 0 – High-income collectors** (high WTP, strong resale interest).",
            1: "🟢 **Cluster 1 – Conscious curators** (high sustainability, moderate WTP).",
            2: "🔵 **Cluster 2 – Hype sneaker owners** (many sneakers, good WTP).",
            3: "🟡 **Cluster 3 – Value seekers** (price sensitive, lower WTP).",
            4: "🟠 **Cluster 4 – Occasional restorers** (few items, medium WTP).",
            5: "⚫ **Cluster 5 – Low-engagement segment** (low WTP, low resale interest).",
        }

        chip_lines = [persona_labels[c] for c in available_clusters if c in persona_labels]

        if chip_lines:
            st.markdown("**Suggested personas (for your report / slides):**")
            st.markdown("\n".join(chip_lines))

        # Build simple text report for download
        lines = [
            "ATELIER 8 – Customer Segment Summary",
            "------------------------------------",
            "",
        ]
        for c in available_clusters:
            row = summary.loc[c]
            lines.append(
                f"Cluster {c}: "
                f"Avg WTP restoration = AED {row['wtp_restoration_aed']:.0f}, "
                f"Avg WTP authentication = AED {row['wtp_authentication_aed']:.0f}, "
                f"Sustainability importance = {row['sustainability_importance']:.1f}, "
                f"Handbags owned = {row['handbags_owned']:.1f}, "
                f"Sneakers owned = {row['sneakers_owned']:.1f}, "
                f"Resale interest = {row['resale_interest']:.1f}."
            )

        report_text = "\n".join(lines)
        st.download_button(
            "Download cluster persona summary (TXT)",
            data=report_text.encode("utf-8"),
            file_name="atelier8_customer_segments_summary.txt",
            mime="text/plain",
        )

    # ---------- TAB 3: MODELS ----------
    with tab_models:
        st.subheader("3️⃣ Predictive Models")

        st.info(
            "These models show how ATELIER 8 can move from descriptive insights (what is happening) "
            "to predictive intelligence (what is likely to happen next)."
        )

        tabA, tabB, tabC = st.tabs(
            ["🧮 Classification", "💰 Regression", "🧷 Associations"]
        )

        # CLASSIFICATION
        with tabA:
            st.markdown("### 🧮 Logistic Regression – Who adopts?")

            st.markdown(
                """
                **What is Logistic Regression and why are we using it?**  
                - Logistic Regression is a **classification algorithm** that predicts the probability of an event with two outcomes (e.g., adopt vs. not adopt).  
                - Here, we predict whether a customer’s adoption score is **high (≥4)** based on their profile (income, ownership, sustainability, resale interest, etc.).  
                - For ATELIER 8, this helps prioritise **which leads are most likely to say “yes” first**, so marketing can focus effort on high-potential customers.
                """
            )

            df_c = filtered_df.copy()
            df_c["adopt"] = (df_c["adoption_intent"] >= 4).astype(int)

            X = pd.get_dummies(
                df_c[[
                    "age_group", "income_level", "gender",
                    "sustainability_importance", "owns_luxury_items",
                    "handbags_owned", "sneakers_owned", "resale_interest"
                ]],
                drop_first=True
            )
            y = df_c["adopt"]

            Xtr, Xts, ytr, yts = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xts)

            acc = accuracy_score(yts, pred)
            st.write(f"**Accuracy on test data:** `{acc:.2f}`")

            report_df = pd.DataFrame(classification_report(yts, pred, output_dict=True)).transpose().round(2)
            st.markdown("**Model performance by class**")
            st.dataframe(report_df)

        # REGRESSION (with scenario simulator)
        with tabB:
            st.markdown("### 💰 Regression Model – What drives WTP for Restoration?")

            st.markdown(
                """
                **What is Regression and why are we using it?**  
                - Regression predicts a **continuous numeric value**, such as willingness to pay (in AED) for a service.  
                - Our Linear Regression model estimates WTP based on income level, number of items, sustainability importance and resale interest.  
                - For ATELIER 8, this tells us **how much different types of customers are likely to pay**, supporting smarter price band design (entry, standard, premium tiers).
                """
            )

            df_r = filtered_df.copy()
            Xr = pd.get_dummies(
                df_r[[
                    "income_level",
                    "sustainability_importance",
                    "handbags_owned",
                    "sneakers_owned",
                    "resale_interest"
                ]],
                drop_first=True
            )
            yr = df_r["wtp_restoration_aed"]

            reg = LinearRegression()
            reg.fit(Xr, yr)

            coef_df = pd.DataFrame(
                {"feature": Xr.columns, "coefficient": reg.coef_}
            ).sort_values("coefficient", ascending=False)
            st.markdown("**Model Coefficients (AED impact per unit change):**")
            st.dataframe(coef_df)

            st.markdown("### Scenario Simulator")

            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                sim_income = st.selectbox("Income level", ["Low", "Mid", "High", "Very High"], index=2)
                sim_sust = st.slider("Sustainability importance", 1, 5, 4)
            with col_sim2:
                sim_bags = st.slider("Handbags owned", 0, 10, 2)
                sim_sneakers = st.slider("Sneakers owned", 0, 10, 3)
            with col_sim3:
                sim_resale = st.slider("Resale interest", 1, 5, 4)

            sim_row = pd.DataFrame(
                {
                    "income_level": [sim_income],
                    "sustainability_importance": [sim_sust],
                    "handbags_owned": [sim_bags],
                    "sneakers_owned": [sim_sneakers],
                    "resale_interest": [sim_resale],
                }
            )

            sim_enc = pd.get_dummies(sim_row, columns=["income_level"], drop_first=True)
            sim_enc = sim_enc.reindex(columns=Xr.columns, fill_value=0)

            pred_wtp = reg.predict(sim_enc)[0]
            st.success(f"Estimated WTP for restoration: **AED {pred_wtp:,.0f}**")

            st.markdown("**Managerial takeaway:**")
            st.write(
                "Use this simulator to test pricing for **High / Very High income** collectors vs entry-level clients. "
                "If Very High income with high sustainability and resale interest yields a much higher WTP, "
                "ATELIER 8 can confidently design **premium tiers** without fear of underpricing."
            )

        # ASSOCIATIONS
        with tabC:
            st.markdown("### 🧷 Brand Association Patterns")

            st.markdown(
                """
                **What is Association Rule Mining and why are we using it?**  
                - Association rule mining looks for **items that frequently occur together** in the same basket or customer profile.  
                - Here, we check which luxury brands (Chanel, LV, Gucci, sneakers, etc.) tend to be owned together by the same person.  
                - For ATELIER 8, this helps design **cross-sell bundles and combined spa offers**, e.g., “Chanel flap + LV Neverfull restoration combo” for customers likely to own both.
                """
            )

            brands = ["owns_chanel", "owns_lv", "owns_dior", "owns_gucci", "owns_hermes", "owns_sneaker_grails"]
            bdf = filtered_df[brands]
            total = len(bdf)

            rules = []
            for i, b1 in enumerate(brands):
                for j, b2 in enumerate(brands):
                    if i >= j:
                        continue
                    both = ((bdf[b1] == 1) & (bdf[b2] == 1)).sum()
                    if both == 0:
                        continue
                    support = both / total
                    conf = both / bdf[b1].sum()
                    rules.append(
                        {
                            "rule": f"{b1} → {b2}",
                            "support": round(support, 3),
                            "confidence": round(conf, 3),
                        }
                    )

            rules_df = pd.DataFrame(rules).sort_values("confidence", ascending=False)
            st.dataframe(rules_df)

    # ---------- TAB 4: UPLOAD ----------
    with tab_upload:
        st.subheader("4️⃣ Upload & Score New Leads")

        st.markdown("**Required columns in your CSV:**")
        st.code(
            "age_group, income_level, gender, sustainability_importance, "
            "owns_luxury_items, handbags_owned, sneakers_owned, resale_interest"
        )

        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            udf = pd.read_csv(file)
            st.write("Preview of uploaded leads:")
            st.dataframe(udf.head())

            # Train classifier on full data
            df_c = df.copy()
            df_c["adopt"] = (df_c["adoption_intent"] >= 4).astype(int)

            Xb = pd.get_dummies(
                df_c[[
                    "age_group", "income_level", "gender",
                    "sustainability_importance", "owns_luxury_items",
                    "handbags_owned", "sneakers_owned", "resale_interest"
                ]],
                drop_first=True
            )
            yb = df_c["adopt"]

            clf_full = LogisticRegression(max_iter=1000)
            clf_full.fit(Xb, yb)

            X_user = pd.get_dummies(
                udf[[
                    "age_group", "income_level", "gender",
                    "sustainability_importance", "owns_luxury_items",
                    "handbags_owned", "sneakers_owned", "resale_interest"
                ]],
                drop_first=True
            )
            X_user = X_user.reindex(columns=Xb.columns, fill_value=0)

            probs = clf_full.predict_proba(X_user)[:, 1]
            udf["adoption_probability"] = np.round(probs, 3)

            st.success("Lead scoring complete.")
            st.dataframe(udf.head())

            st.download_button(
                "Download scored CSV",
                udf.to_csv(index=False).encode("utf-8"),
                "atelier8_scored_leads.csv",
                "text/csv"
            )

    st.markdown("---")
    st.caption("ATELIER 8 · Circular Luxury · Data-Driven Stewardship")


if __name__ == "__main__":
    main()
